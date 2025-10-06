# OpenAI Reasoning API 适配器

这是一个 FastAPI 代理服务，用于将标准的 OpenAI Chat Completion 请求转换为新的 Reasoning API 格式，且保留推理摘要。。

## 功能特性

- 🚀 将 OpenAI Chat Completion 请求转换为 Reasoning API 格式
- 🔄 支持流式和非流式响应
- 🔐 转发认证头部到后端 API
- 📡 自动处理响应格式转换
- 🐳 支持 Docker 部署
- ⚙️ 可配置的后端 API URL
- 🧠 保留并展示推理过程和摘要

## 快速开始

### 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行服务：
```bash
python main.py
```

或者使用 uvicorn：
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker 运行

1. 构建镜像：
```bash
docker build -t openai-reasoning-proxy .
```

2. 运行容器：
```bash
docker run -p 8000:8000 openai-reasoning-proxy
```

3. 使用环境变量自定义配置：
```bash
docker run -p 8000:8000 -e BACKEND_API_URL="https://your-api-endpoint.com/v1/responses" openai-reasoning-proxy
```

## API 使用

### 端点
- `POST /v1/chat/completions` - 主要的代理端点

### 请求格式

发送标准的 OpenAI Chat Completion 请求格式，例如：

```json
{
  "model": "gpt-5",
  "messages": [
    {"role": "user", "content": "你好，请帮我解释一下量子计算的基本原理。"}
  ],
  "stream": true,
  "temperature": 0.7
}
```

### 响应格式

服务会返回 OpenAI 兼容的响应格式，包括推理过程：

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1699297661,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<thinking>\n让我先理解用户的问题...\n</thinking>\n量子计算的基本原理是..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## 配置

### 环境变量

- `BACKEND_API_URL` - 后端 Reasoning API 的 URL（默认：`https://api.openai.com/v1/responses`）

### 请求参数

服务支持以下 OpenAI Chat Completion 参数：
- `model`
- `stream`
- `temperature`
- `top_p`
- `max_tokens`
- `user`
- `metadata`
- `reasoning_effort`
- `reasoning_summary`

## 开发

### 项目结构

```
├── main.py              # 主要的 FastAPI 应用
├── requirements.txt     # Python 依赖
├── Dockerfile          # Docker 配置文件
└── README.md           # 项目说明文档
```

### 测试

服务启动后，可以使用 curl 进行测试：

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## 许可证

MIT License