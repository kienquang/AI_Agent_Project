# core/state.py
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# 1. State của toàn bộ chu trình Graph
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    pending_ticket: dict
    summary: str

# 2. Khuôn Pydantic ép LLM trả dữ liệu chuẩn
class TicketData(BaseModel):
    name: str = Field(description="Tên của khách hàng. Bắt buộc để trống '' nếu khách chưa xưng tên.")
    email: str = Field(description="Email khách hàng. Trống '' nếu chưa cung cấp.")
    issue: str = Field(description="Vấn đề khiếu nại của khách. Để trống '' nếu chưa rõ.")