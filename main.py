import os
import json
import time
import hmac
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="AI 超级工作台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 服务端环境变量配置
DEFAULT_API_KEY = os.getenv("AI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("AI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
DEFAULT_MODEL = os.getenv("AI_MODEL", "deepseek-chat")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "15000"))

# 轻量级内存 IP 限流器
_ip_requests: Dict[str, List[float]] = defaultdict(list)


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(ip: str):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    window_start = now - 60.0
    timestamps = [t for t in _ip_requests[ip] if t > window_start]
    if len(timestamps) >= RATE_LIMIT_PER_MIN:
        _ip_requests[ip] = timestamps
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，请稍后再试（限制每分钟 {RATE_LIMIT_PER_MIN} 次）"
        )
    timestamps.append(now)
    _ip_requests[ip] = timestamps

    # 定期清理过期 IP 记录，防止内存增长
    if len(_ip_requests) > 3000:
        expired = [k for k, v in _ip_requests.items() if not v or v[-1] <= window_start]
        for k in expired:
            del _ip_requests[k]


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    access_password: Optional[str] = ""
    custom_api_key: Optional[str] = ""
    custom_base_url: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """云平台健康检查端点"""
    return {"status": "ok"}


@app.get("/api/config")
async def get_public_config():
    return {
        "has_server_key": bool(DEFAULT_API_KEY),
        "requires_password": bool(ACCESS_PASSWORD),
        "default_model": DEFAULT_MODEL,
        "default_base_url": DEFAULT_BASE_URL,
        "max_input_chars": MAX_INPUT_CHARS,
    }


@app.post("/api/chat")
async def chat_stream(req: ChatRequest, request: Request):
    # 1. IP 限流检查
    client_ip = get_client_ip(request)
    check_rate_limit(client_ip)

    # 2. 密码鉴权（使用恒定时间比对）
    use_custom_key = bool(req.custom_api_key and req.custom_api_key.strip())
    if ACCESS_PASSWORD and not use_custom_key:
        provided = (req.access_password or "").encode("utf-8")
        expected = ACCESS_PASSWORD.encode("utf-8")
        if not hmac.compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="访问密码错误，请在右上角设置中填写正确密码")

    # 3. 清洗消息并限制长度（最多保留最近 20 条上下文）
    clean_messages = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in req.messages[-20:]
        if m.get("role") in {"system", "user", "assistant"} and str(m.get("content", "")).strip()
    ]
    if not clean_messages:
        raise HTTPException(status_code=400, detail="输入内容不能为空")

    total_chars = sum(len(m["content"]) for m in clean_messages)
    if total_chars > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"上下文总字数超出限制（当前 {total_chars} 字，上限 {MAX_INPUT_CHARS} 字），请缩短内容或清空对话"
        )

    # 4. 解析模型配置
    api_key = req.custom_api_key.strip() if use_custom_key else DEFAULT_API_KEY
    base_url = (req.custom_base_url.strip() if req.custom_base_url else DEFAULT_BASE_URL).rstrip("/")
    model = req.model.strip() if (req.model and req.model.strip()) else DEFAULT_MODEL

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="未配置 API Key！请在服务器环境变量设置 AI_API_KEY，或在设置中填入你自己的 Key。"
        )

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": clean_messages,
        "temperature": req.temperature,
        "stream": True,
    }

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        err_bytes = await resp.aread()
                        err_text = err_bytes.decode("utf-8", errors="ignore")
                        yield f"data: {json.dumps({'error': f'上游模型接口报错 ({resp.status_code}): {err_text}'}, ensure_ascii=False)}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if line and line.startswith("data: "):
                            yield f"{line}\n\n"
        except httpx.RequestError as e:
            yield f"data: {json.dumps({'error': f'上游网络请求失败: {str(e)}'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'服务器内部错误: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)