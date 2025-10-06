import os
import json
import time
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any, AsyncGenerator

# --- Configuration ---
# The URL of the backend Reasoning API.
# It can be overridden by setting the BACKEND_API_URL environment variable.
BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL", "https://api.openai.com/v1/responses")

app = FastAPI(
    title="OpenAI Reasoning API Adapter",
    description="A proxy to convert OpenAI Chat Completion requests to a new Reasoning API format.",
    version="1.0.6"
)


def translate_to_responses_format(chat_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates a standard OpenAI Chat Completion request body to the
    backend's Reasoning API format.
    """
    backend_payload = {}
    last_user_message = ""
    # Find the last user message from the 'messages' array.
    if "messages" in chat_request and isinstance(chat_request["messages"], list):
        for message in reversed(chat_request["messages"]):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                last_user_message = message["content"]
                break
    if not last_user_message:
        raise ValueError("No valid user message found in the request.")

    backend_payload["input"] = last_user_message

    # Pass through common parameters from the original request.
    passthrough_params = [
        "model", "stream", "temperature", "top_p", "max_tokens",
        "user", "metadata"
    ]
    for param in passthrough_params:
        if param in chat_request:
            backend_payload[param] = chat_request[param]

    # Handle specific reasoning parameters.
    reasoning_params = {}
    if "reasoning_effort" in chat_request:
        reasoning_params["effort"] = chat_request["reasoning_effort"]
    reasoning_params["summary"] = chat_request.get("reasoning_summary", "auto")
    if reasoning_params:
        backend_payload["reasoning"] = reasoning_params

    return backend_payload


def format_as_chat_completion_chunk(id: str, model: str, content: str, role: str = None) -> str:
    """
    Formats a piece of data into a server-sent event (SSE) string
    that mimics the OpenAI Chat Completion streaming chunk format.
    """
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


async def stream_generator(backend_response: httpx.Response) -> AsyncGenerator[str, None]:
    """
    Parses the streaming response from the backend Reasoning API and yields
    OpenAI-compatible chat completion chunks.
    """
    reasoning_accumulator = ""
    # 新增: 用于存储已收到的完整输出文本
    full_output_accumulator = ""
    is_reasoning_sent = False
    is_first_content_chunk = True  # 新增: 用于判断是否是第一个内容块
    response_id = "unknown-stream-id"
    model_name = "unknown-model"
    buffer = ""
    done = False

    try:
        async for chunk in backend_response.aiter_raw():
            buffer += chunk.decode("utf-8", errors="ignore")
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                data_str = line.split("data:", 1)[1].lstrip()
                if data_str == "[DONE]":
                    done = True
                    break

                try:
                    event_data = json.loads(data_str)
                    event_type = event_data.get("type")

                    if event_type == "response.created":
                        response_id = event_data.get(
                            "response", {}).get("id", response_id)
                        model_name = event_data.get(
                            "response", {}).get("model", model_name)

                    elif event_type == "response.reasoning_summary_text.delta":
                        reasoning_accumulator += event_data.get("delta", "")

                    elif event_type in ("response.reasoning_summary_part.done", "response.reasoning_summary_text.done"):
                        if not is_reasoning_sent and reasoning_accumulator.strip():
                            thinking_block = f"<thinking>\n{reasoning_accumulator.strip()}\n</thinking>\n"
                            yield format_as_chat_completion_chunk(response_id, model_name, thinking_block, role="assistant")
                            is_reasoning_sent = True
                            is_first_content_chunk = False  # "thinking"块发出后，下一个内容块就不是第一个了

                    elif event_type == "response.output_text.delta":
                        # --- START: 这是核心修改部分 ---
                        new_full_content = event_data.get("delta", "")
                        # 计算真正的增量
                        actual_delta = new_full_content[len(
                            full_output_accumulator):]
                        # 更新累加器
                        full_output_accumulator = new_full_content

                        if actual_delta:
                            # 只有在第一个内容块时才发送 "assistant" 角色
                            role = "assistant" if is_first_content_chunk else None
                            yield format_as_chat_completion_chunk(response_id, model_name, actual_delta, role=role)
                            if role:
                                is_first_content_chunk = False  # 发送后就不是第一个了
                        # --- END: 核心修改部分结束 ---

                    elif event_type in ("response.completed", "response.cancelled", "response.error"):
                        done = True
                        break

                except json.JSONDecodeError:
                    continue

            if done:
                break

    except httpx.ReadError as e:
        print(
            f"An httpx.ReadError occurred: {e}. This may happen if the client disconnects.")
    finally:
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    """
    The main proxy endpoint. It receives an OpenAI Chat Completion request,
    transforms it, sends it to the backend, and then transforms the response back.
    """
    # Get the Authorization header from the incoming request.
    # This header should contain the API key (e.g., "Bearer sk-...").
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
    # Set a long timeout, especially for streaming responses.
    timeout_config = httpx.Timeout(300.0, read=None)

    # --- Streaming Logic ---
    if is_stream:
        client = httpx.AsyncClient(timeout=timeout_config)
        headers = {
            "Authorization": auth_header,  # Forward the original auth header.
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            backend_req = client.build_request(
                "POST", BACKEND_API_URL, json=backend_payload, headers=headers)
            response = await client.send(backend_req, stream=True)
            # Raise an exception for 4xx/5xx responses.
            response.raise_for_status()

        except httpx.RequestError as e:
            await client.aclose()
            raise HTTPException(
                status_code=502, detail=f"Could not connect to backend API: {e}")
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            await e.response.aclose()
            await client.aclose()
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Error from backend API: {error_content.decode()}")

        async def downstream():
            """Ensures all resources are properly closed after streaming."""
            try:
                async for chunk in stream_generator(response):
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            downstream(),
            media_type="text/event-stream"
        )

    # --- Non-Streaming Logic ---
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        headers = {
            "Authorization": auth_header,  # Forward the original auth header.
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = await client.post(
                BACKEND_API_URL, json=backend_payload, headers=headers)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502, detail=f"Could not connect to backend API: {e}")
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Error from backend API: {error_content.decode()}")

        backend_data = response.json()
        reasoning_summary = ""
        message_content = ""

        # Parse the non-streaming response structure.
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

        # Assemble the final OpenAI-compatible response.
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
    # To run this script, use the command: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
