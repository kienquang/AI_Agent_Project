# core/engine.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Import từ các file được tách ra
from core.state import AgentState
from core.nodes import (
    guard_node, memory_manager_node, supervisor_node, 
    rag_agent_node, prepare_ticket_node, execute_ticket_node,
    guard_router, route_logic, route_after_prepare
)

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# 1. KẾT NỐI DATABASE & MEMORY
pool = ConnectionPool(conninfo=DB_URL, max_size=10, min_size=1, kwargs={"autocommit": True})
memory = PostgresSaver(pool)
memory.setup()

# 2. VẼ SƠ ĐỒ GRAPH
workflow = StateGraph(AgentState)

workflow.add_node("GUARD", guard_node)
workflow.add_node("MEMORY", memory_manager_node)
workflow.add_node("SUPERVISOR", supervisor_node)
workflow.add_node("RAG_Agent", rag_agent_node)
workflow.add_node("Prepare_Ticket", prepare_ticket_node)
workflow.add_node("Execute_Ticket", execute_ticket_node)

workflow.add_edge(START, "GUARD")
workflow.add_conditional_edges(
    "GUARD",
    guard_router,
    {
        "MEMORY": "MEMORY",  # Sạch -> Đẩy vào trong cho Supervisor
        "FINISH": END                # Bẩn -> Đuổi ra ngoài (KẾT THÚC)
    }
)

# Từ Memory đi thẳng sang Supervisor
workflow.add_edge("MEMORY", "SUPERVISOR")

workflow.add_conditional_edges(
    "SUPERVISOR",
    route_logic,
    {
        "RAG": "RAG_Agent",
        "ACTION": "Prepare_Ticket",
        "FINISH": END
    }
)

workflow.add_edge("RAG_Agent", "SUPERVISOR")  # Sau khi RAG trả lời, quay lại Supervisor để kiểm tra nếu cần
# Sau khi chuẩn bị xong -> Bắt buộc rẽ vào Thực thi
workflow.add_conditional_edges(
    "Prepare_Ticket",
    route_after_prepare,
    {
        "Execute_Ticket": "Execute_Ticket",
        "SUPERVISOR": "SUPERVISOR"
    }
)
workflow.add_edge("Execute_Ticket", "SUPERVISOR")

# Biên dịch Graph
app_graph = workflow.compile(checkpointer=memory, interrupt_before=["Execute_Ticket"])

# 3. CỔNG GIAO TIẾP VỚI FASTAPI
def process_chat_messages(user_query: str, session_id: str, user_action: str = "chat"):
    """
    - user_query: Câu hỏi mới nhất.
    - session_id: Mã phiên chat (Để DB biết phải lục lại trí nhớ của ai).
    """
    # Cấu hình thread_id để lấy lại trí nhớ
    config = {"configurable": {"thread_id": session_id}}

    # Biên dịch và cài đặt điểm dừng
    app_graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["Execute_Ticket"] #Dừng lại trước khi chạy node này
    )

    # Lấy trạng thái hiện tại của Graph
    state = app_graph.get_state(config)

    # Nếu Graph bị dừng ở state execute_ticket chờ xác nhận
    if "Execute_Ticket" in state.next:
        if user_action == "approve":
            print("\n✅ Khách hàng ĐỒNG Ý. Cấp quyền chạy tiếp...", flush=True)
            # Dùng None để Graph tự động chạy tiếp từ chỗ nó đang dừng
            response_state = app_graph.invoke(None, config=config)

        elif user_action == "reject":
            print("\n❌ Khách hàng HỦY. Xóa dữ liệu tạm.", flush=True)
            # CẬP NHẬT TRẠNG THÁI VÀ "GIẢ DANH" NODE ĐỂ BỎ QUA NÓ
            app_graph.update_state(
                config, 
                {
                    "pending_ticket": None,
                    # Thêm thẳng câu trả lời của AI vào luôn để báo khách biết
                    "messages": [AIMessage(content="Dạ, em đã hủy yêu cầu tạo phiếu theo ý anh/chị ạ. Anh/chị cần hỗ trợ gì thêm không?")]
                },
                as_node="Execute_Ticket" # TỪ KHÓA QUAN TRỌNG NHẤT
            )
            
            # Sau khi đã bypass thành công, gọi lệnh invoke để hệ thống 
            # chạy nốt chu trình về đích (Supervisor -> FINISH)
            response_state = app_graph.invoke(None, config=config)

    else:
        # Nếu chạy bình thường (Chat)
        response_state = app_graph.invoke(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config
        )
    # Kiểm tra xem SAU KHI chạy xong vòng này, nó có bị DỪNG lại không?
    new_state = app_graph.get_state(config)
    requires_confirmation = "Execute_Ticket" in new_state.next
    
    return {
        "reply": response_state["messages"][-1].content,
        "requires_confirmation": requires_confirmation
    }
