from fastapi import FastAPI
from pydantic import BaseModel
import time

# import bộ não từ core
from core.agent_workflow import process_chat_message

app = FastAPI(title = "ABC Company - AI Agent API")

# Định nghĩa dữ liệu
class Message(BaseModel):
    role: str
    content: str

class ChatCompleteRequest(BaseModel):
    model: str
    messages: list[Message]

# API endpoint
@app.get("/v1/models")
async def get_models():
    """Cung cấp danh sách Model cho Open WebUI"""
    return {
        "object": "list",
        "data": [
            {"id": "llama-3.1-8b-instant", "object": "model", "created": int(time.time()), "owned_by": "custom"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_endpoint(request: ChatCompleteRequest):
    """Điểm tiếp nhận tin nhắn từ giao diện, ném cho Core xử lý và trả về"""

    # gọi hàm xử lý chính trong core, nhận về câu trả lời
    final_answer = process_chat_message(request.messages)

    # trả về kết quả theo chuẩn openapi
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_answer},
                "finish_reason": "stop"
            }
        ]
    }
