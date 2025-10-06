import os
import json
import time
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, AsyncGenerator

# --- Configuration ---
BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL", "https://api.openai.com/v1/responses")

app = FastAPI(
    title="OpenAI Reasoning API Adapter",
    description="A proxy to convert OpenAI Chat Completion requests to the new Reasoning API format.",
    version="1.0.5-robust-streaming"  # Version bump for the robust streaming fix
)

# ... (translate_to_responses_format and format_as_chat_completion_chunk remain the same) ...


def translate_to_responses_format(chat_request: Dict[str, Any]) -> Dict[str, Any]:
    backend_payload = {}
    last_user_message = ""
    if "messages" in chat_request and isinstance(chat_request["messages"], list):
        for message in reversed(chat_request["messages"]):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                last_user_message = message["content"]
                break
    if not last_user_message:
        raise ValueError("No valid user message found in the request.")
    backend_payload["input"] = last_user_message
    passthrough_params = [
        "model", "stream", "temperature", "top_p", "max_tokens",
        "user", "metadata"
    ]
    for param in passthrough_params:
        if param in chat_request:
            backend_payload[param] = chat_request[param]
    reasoning_params = {}
    if "reasoning_effort" in chat_request:
        reasoning_params["effort"] = chat_request["reasoning_effort"]
    reasoning_params["summary"] = chat_request.get("reasoning_summary", "auto")
    if reasoning_params:
        backend_payload["reasoning"] = reasoning_params
    return backend_payload


def format_as_chat_completion_chunk(id: str, model: str, content: str, role: str = None) -> str:
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": None
            }
        ]
    }
    if role:
        chunk["choices"][0]["delta"]["role"] = role
    if content:
        chunk["choices"][0]["delta"]["content"] = content
    return f"data: {json.dumps(chunk)}\n\n"


async def stream_generator(backend_response: httpx.Response):
    reasoning_accumulator = ""
    is_reasoning_sent = False
    response_id = "unknown"
    model_name = "unknown"
    buffer = ""
    done = False
    try:
        async for chunk in backend_response.aiter_raw():
            buffer += chunk.decode("utf-8", errors="ignore")
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.rstrip('\r')

                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line.split("data:", 1)[1].lstrip()
                if data_str == "[DONE]":
                    done = True
                    break

                try:
                    event_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event_data.get("type")

                if event_type == "response.created":
                    response_id = event_data.get(
                        "response", {}).get("id", response_id)
                    model_name = event_data.get(
                        "response", {}).get("model", model_name)

                elif event_type == "response.reasoning_summary_text.delta":
                    reasoning_accumulator += event_data.get("delta", "")

                # --- FIX IS HERE ---
                elif event_type in ("response.reasoning_summary_part.done", "response.reasoning_summary_text.done"):
                    if not is_reasoning_sent and reasoning_accumulator.strip():
                        thinking_block = f"<thinking>\n{reasoning_accumulator.strip()}\n</thinking>\n"
                        yield format_as_chat_completion_chunk(response_id, model_name, thinking_block, role="assistant")
                        is_reasoning_sent = True
                # --- END OF FIX ---

                elif event_type == "response.output_text.delta":
                    delta_content = event_data.get("delta", "")
                    if delta_content:
                        # Only send role="assistant" on the very first chunk sent
                        role = "assistant" if not is_reasoning_sent else None
                        yield format_as_chat_completion_chunk(response_id, model_name, delta_content, role=role)
                        if role:  # If we just sent the role, mark it as sent for all future chunks
                            is_reasoning_sent = True

                elif event_type in ("response.completed", "response.cancelled", "response.error"):
                    done = True
                    break

            if done:
                break

    except httpx.ReadError as e:
        print(f"ReadError during raw stream processing, treating as EOF: {e}")

    # Send the final DONE signal
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401, detail="Authorization header is required.")

    try:
        chat_request_json = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    try:
        backend_payload = translate_to_responses_format(chat_request_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_stream = backend_payload.get("stream", False)

    # 提高 read 超时时间，SSE 常见
    timeout_config = httpx.Timeout(300.0, read=None)

    if is_stream:
        # 不要用 async with，这样客户端会在返回后立刻被关闭
        client = httpx.AsyncClient(timeout=timeout_config)
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            backend_req = client.build_request(
                "POST", BACKEND_API_URL, json=backend_payload, headers=headers)
            response = await client.send(backend_req, stream=True)

            if response.status_code != 200:
                error_content = await response.aread()
                await response.aclose()
                await client.aclose()
                raise HTTPException(status_code=response.status_code,
                                    detail=f"Error from backend API: {error_content}")

        except httpx.RequestError as e:
            await client.aclose()
            raise HTTPException(
                status_code=502, detail=f"Could not connect to backend API: {e}")

        async def downstream():
            try:
                async for chunk in stream_generator(response):
                    yield chunk
            finally:
                # 确保资源在流结束后关闭
                try:
                    await response.aclose()
                finally:
                    await client.aclose()

        return StreamingResponse(
            downstream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # 非流式：维持原逻辑，用 async with 即可
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            backend_req = client.build_request(
                "POST", BACKEND_API_URL, json=backend_payload, headers=headers)
            response = await client.send(backend_req)
            if response.status_code != 200:
                error_content = await response.aread()
                raise HTTPException(status_code=response.status_code,
                                    detail=f"Error from backend API: {error_content}")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502, detail=f"Could not connect to backend API: {e}")

        backend_data = response.json()
        reasoning_summary = ""
        message_content = ""
        for item in backend_data.get("output", []):
            if item.get("type") == "reasoning":
                for summary_part in item.get("summary", []):
                    if summary_part.get("type") == "summary_text":
                        reasoning_summary += summary_part.get("text", "")
            elif item.get("type") == "message":
                for content_part in item.get("content", []):
                    if content_part.get("type") == "output_text":
                        message_content += content_part.get("text", "")

        final_content = ""
        if reasoning_summary.strip():
            final_content += f"<thinking>\n{reasoning_summary.strip()}\n</thinking>\n"
        final_content += message_content.strip()

        final_response = {
            "id": backend_data.get("id"),
            "object": "chat.completion",
            "created": backend_data.get("created_at"),
            "model": backend_data.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_content,
                },
                "finish_reason": "stop"
            }],
            "usage": backend_data.get("usage")
        }
        return JSONResponse(content=final_response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
