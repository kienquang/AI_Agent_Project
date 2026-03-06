import httpx
from typing import Annotated, Literal
# from streamlit import json
import json as json_lib
import re
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Nhập RAG
from core.rag_engine import retrieve_context

load_dotenv()

# 1. Định nghĩa state (Bộ nhớ của đoạn chat)
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str  #Lưu quyết định của supervisor

# 2. Khởi tạo LLM và công cụ
llm = ChatGroq(model="llama-3.1-8b-instant")
N8N_WEBHOOK_URL = "https://hook.eu1.make.com/hiklukm85va8ikrm3x9cciql6ss4fvgr"

# 3. Định nghĩa các Agent (NODES)
def supervisor_node(state: AgentState):
    """Quản lý: Phân tích câu hỏi và quyết định chuyển cho ai"""
    print("👔 [Supervisor] Đang phân tích yêu cầu...")
    messages = state["messages"]
    last_message = messages[-1]  # Lấy câu hỏi mới nhất của user

    # CỰC KỲ QUAN TRỌNG: Nếu người vừa nói là AI, nghĩa là đã làm xong việc -> KẾT THÚC
    if isinstance(last_message, AIMessage):
        print("👔 [Supervisor] AI đã trả lời xong, chốt FINISH.", flush=True)
        return {"next_agent": "FINISH"}

    user_query = last_message.content
    print(f"🗣️ Khách nói: {user_query}", flush=True)

    # DÙNG FEW-SHOT PROMPTING ĐỂ ÉP KHUÔN LLM
    prompt = f"""
    Bạn là bộ định tuyến (Router) của tổng đài. Phân loại tin nhắn của khách hàng.
    Chỉ trả về 1 từ duy nhất: "ACTION" hoặc "RAG". Không giải thích.

    Quy tắc & Ví dụ:
    1. Khách phàn nàn, báo lỗi, đòi tạo ticket/phiếu, khiếu nại -> Trả về: ACTION
    2. Khách cung cấp tên CỦA HỌ để TẠO PHIẾU sửa chữa -> Trả về: ACTION
    3. Khách hỏi chính sách, hỏi bâng quơ, chào hỏi -> Trả về: RAG

    Tin nhắn của khách: "{user_query}"
    """
    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().upper()

    # Mặc định là an toàn
    if "ACTION" in decision:
        next_step = "ACTION"
    else:
        next_step = "RAG"

    print(f"👔 [Supervisor] Quyết định rẽ nhánh -> {next_step}")
    return {"next_agent": next_step}

def rag_agent_node(state: AgentState):
    """Chuyên viên Nội quy: Lấy context và trả lời"""
    print("📚 [RAG Agent] Đang tra cứu tài liệu và trả lời...")
    messages = state["messages"]
    user_query = messages[-1].content

    context = retrieve_context(user_query)

    prompt = f"""
    Bạn là nhân viên tư vấn của Công ty ABC. TUYỆT ĐỐI tuân thủ 2 quy tắc sau:
    1. CHỈ sử dụng thông tin trong [TÀI LIỆU] dưới đây để trả lời.
    2. NẾU TÀI LIỆU KHÔNG CÓ THÔNG TIN (ví dụ: khách hỏi tuyển dụng, giá cả không có trong tài liệu), BẮT BUỘC phải trả lời: "Xin lỗi, hiện tại em chưa có thông tin về vấn đề này. Anh/chị vui lòng để lại thông tin hoặc gọi Hotline 1900-xxxx ạ."
    
    KHÔNG ĐƯỢC TỰ BỊA ĐẶT THÔNG TIN.

    [TÀI LIỆU]: {context}
    """

    # Truyền cả lịch sử chat vào để AI nhớ ngữ cảnh
    response = llm.invoke([SystemMessage(content=prompt)]+ messages)
    return {"messages": [response]}

def action_agent_node(state: AgentState):
    """Chuyên viên Xử lý: Gọi Webhook"""
    print("🛠️ [Action Agent] Đang xử lý khiếu nại/tạo Ticket...")
    messages = state["messages"]

    # Để đơn giản, ta nhờ LLM trích xuất tên và vấn đề từ lịch sử chat
    extract_prompt = """
    Nhiệm vụ của bạn là đọc lịch sử hội thoại và trích xuất dữ liệu.
    BẠN LÀ MỘT CỖ MÁY XUẤT DỮ LIỆU. BẠN BỊ CẤM GIAO TIẾP NHƯ CON NGƯỜI.
    BẮT BUỘC CHỈ ĐƯỢC PHÉP TRẢ VỀ DUY NHẤT 1 KHỐI DỮ LIỆU CÓ DẤU { } BAO QUANH.
    
    QUY TẮC:
    1. Tuyệt đối KHÔNG viết thêm bất kỳ câu chào hỏi, giải thích hay câu hỏi nào.
    2. Nếu khách CHƯA XƯNG TÊN, phần "name" BẮT BUỘC là chuỗi rỗng "". KHÔNG điền "Khách hàng", "Ẩn danh".
    3. Nếu khách CHƯA NÓI RÕ LỖI, phần "issue" BẮT BUỘC là chuỗi rỗng "".

    MẪU BẮT BUỘC PHẢI TUÂN THEO:
    {
        "name": "", 
        "issue": ""
    }
    """
    raw_response = llm.invoke([SystemMessage(content=extract_prompt)] + messages).content.strip()
    print(f"🛠️ [Action Agent] Dữ liệu thô AI trích xuất: {raw_response}", flush=True)

    try:
        # VŨ KHÍ MỚI: Dùng Regex để "quét" từ dấu { đầu tiên đến dấu } cuối cùng
        match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        
        if not match:
            # Nếu AI không trả về bất kỳ dấu ngoặc nhọn nào
            print("❌ [Action Agent] Không tìm thấy dấu { } trong câu trả lời.", flush=True)
            return {"messages": [AIMessage(content="Dạ anh/chị có thể nói rõ hơn về tên và vấn đề để em ghi nhận không ạ?")]}

        # Lấy phần chuỗi JSON nguyên chất đã được lọc
        clean_json_str = match.group(0)
        
        # Phân tích JSON
        data = json_lib.loads(clean_json_str)
        name = data.get("name", "").strip()
        issue = data.get("issue", "").strip()
        # Kiem tra điều kiện
        # Nếu AI không tìm thấy tên, hoặc cố tình bịa ra các chữ như "Tên", "Khách hàng"
        name_lower = name.lower()
        
        # Nếu tên bị trống, HOẶC chứa bất kỳ từ nào trong danh sách cấm
        is_invalid_name = not name or any(bad_word in name_lower for bad_word in ["khách", "tên", "ẩn danh", "chưa", "không", "null", "user"])
        
        if is_invalid_name:
            print("🛠️ [Action Agent] Khách chưa cho tên. Dừng Webhook, quay lại hỏi tên!", flush=True)
            return {"messages": [AIMessage(content="Dạ, để em tạo phiếu hỗ trợ sự cố, anh/chị cho em xin Tên của mình để tiện xưng hô và ghi vào phiếu nhé?")]}
        
        if not issue:
            return {"messages": [AIMessage(content="Dạ anh/chị đang gặp sự cố cụ thể là gì để em ghi chú chi tiết vào phiếu kỹ thuật ạ?")]}
        #  Nếu đã đủ thông tin thì mới gọi webhook tạo ticket
        print(f"🛠️ [Action Agent] Đủ thông tin. Bắn Webhook cho {name}...", flush=True)

        # Gọi webhook để tạo ticket
        response = httpx.post(N8N_WEBHOOK_URL, json={"name": name, "issue": issue}, timeout=5)

        if response.status_code == 200:
            reply = f"Dạ, em đã tạo phiếu hỗ trợ sự cố cho anh/chị **{name}** rồi ạ. Bộ phận kỹ thuật sẽ liên hệ sớm nhất."
        else:
            reply = "Rất tiếc hệ thống tạo phiếu đang bảo trì, anh chị gọi hotline giúp em nhé."
        return {"messages": [AIMessage(content=reply)]}

    except json_lib.JSONDecodeError:
        print("❌ [Action Agent] Lỗi phân tích JSON từ AI.", flush=True)
        return {"messages": [AIMessage(content="Dạ anh/chị có thể nói rõ hơn về tên và vấn đề để em ghi nhận không ạ?")]}
    except Exception as e:
        print(f"❌ [Action Agent] Lỗi mạng: {e}", flush=True)
        return {"messages": [AIMessage(content="Hệ thống đang bận, anh chị vui lòng thử lại sau nhé.")]}

# 4. VẼ SƠ ĐỒ LUỒNG (GRAPH) VÀ GẮN TRÍ NHỚ (MEMORY)
workflow = StateGraph(AgentState)

# Thêm các node
workflow.add_node("SUPERVISOR", supervisor_node)
workflow.add_node("RAG_Agent", rag_agent_node)
workflow.add_node("Action_Agent", action_agent_node)

# Thiết lập đường đi
workflow.add_edge(START, "SUPERVISOR")

# Điều kiện rẽ nhánh từ Supervisor
def route_logic(state:AgentState) ->str:
    # Nếu Supervisor bảo là FINISH, chúng ta kết thúc
    if state["next_agent"] == "FINISH":
        return "FINISH"
    # Nếu không, cứ đi theo lệnh của Supervisor (RAG hoặc ACTION)
    return state["next_agent"]

workflow.add_conditional_edges(
    "SUPERVISOR",
    route_logic,
    {
        "RAG": "RAG_Agent",
        "ACTION": "Action_Agent",
        "FINISH": END
    }
)

workflow.add_edge("RAG_Agent", "SUPERVISOR")  # Sau khi RAG trả lời, quay lại Supervisor để kiểm tra nếu cần
workflow.add_edge("Action_Agent", "SUPERVISOR")  # Sau khi Action_Agent xử lý, quay lại Supervisor để kiểm tra nếu cần

# 5. Hàm giao tiếp cho API
def process_chat_messages(user_query: str, session_id: str):
    """
    - user_query: Câu hỏi mới nhất.
    - session_id: Mã phiên chat (Để DB biết phải lục lại trí nhớ của ai).
    """
    # Cấu hình thread_id để lấy lại trí nhớ
    config = {"configurable": {"thread_id": session_id}}

    # MỞ KẾT NỐI DATABASE AN TOÀN TRONG NGỮ CẢNH CỦA REQUEST NÀY
    # Dùng "with" đảm bảo chạy xong request sẽ tự đóng DB, không bị kẹt!
    with SqliteSaver.from_conn_string("memory.sqlite") as memory:
        # Lắp trí nhớ vào Graph ngay tại thời điểm gọi
        app_graph = workflow.compile(checkpointer=memory)
        
        response_state = app_graph.invoke(
            {"messages": [HumanMessage(content=user_query)]}, 
            config=config
        )

    # Lấy tin nhắn cuối cùng do Rag hoặc Action Agent trả về
    final_answer = response_state["messages"][-1].content
    return final_answer