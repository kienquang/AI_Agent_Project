from fastapi import FastAPI
from pydantic import BaseModel
import time

# import bộ não từ core
from core.agent_workflow import process_chat_messages

app = FastAPI(title = "ABC Company - AI Agent API")

# Định nghĩa dữ liệu
class Message(BaseModel):
    role: str
    content: str

class ChatCompleteRequest(BaseModel):
    model: str
    messages: list[Message]
    user: str = "default_session"
    action: str = "chat" 

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

    # Chỉ cần lấy câu cuối cùng của khách
    user_query = request.messages[-1].content if request.messages else ""
    session_id = request.user 
    action = request.action
    
    # Gọi hàm xử lý và nhận về Dictionary
    result = process_chat_messages(user_query, session_id, action)

    # Trả về Custom Response chứa thông tin cho Giao diện UI vẽ Nút bấm
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0, 
            "message": {
                "role": "assistant", 
                "content": result["reply"],
                "requires_confirmation": result["requires_confirmation"] # Báo cho UI biết phải vẽ 2 cái nút
            }, 
            "finish_reason": "stop"
        }]
    }