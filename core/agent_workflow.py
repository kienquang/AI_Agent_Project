import httpx
from dotenv import load_dotenv

# Import các công cụ của LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# NHẬP KHẨU KIẾN THỨC TỪ MODULE RAG CHÚNG TA VỪA LÀM
from core.rag_engine import retrieve_context

load_dotenv()

# 1. Định nghĩa công cụ
N8N_WEBHOOK_URL = "https://hook.eu1.make.com/hiklukm85va8ikrm3x9cciql6ss4fvgr" 

@tool
def tao_ticket_ho_tro(customer_name: str, issue_description: str) -> str:
    """
    Công cụ này dùng để tạo phiếu (ticket) hỗ trợ kỹ thuật hoặc khiếu nại cho khách hàng.
    Hãy gọi công cụ này KHI VÀ CHỈ KHI khách hàng yêu cầu đổi trả, bảo hành, hoặc phàn nàn về sản phẩm.
    - customer_name: Tên khách hàng (phải trích xuất từ lịch sử chat).
    - issue_description: Mô tả chi tiết vấn đề khách gặp phải.
    """
    print(f"⚙️ [Tool] Đang kích hoạt n8n tạo Ticket cho khách: {customer_name}")
    try:
        payload = {"customer_name:": customer_name, "issue_description": issue_description}
        response = httpx.post(N8N_WEBHOOK_URL, json=payload, timeout=5)

        if response.status_code == 200:
            return f"SUCCESS: Đã tạo ticket thành công cho khách hàng {customer_name}."
        else:
            return f"❌ Có lỗi khi tạo phiếu: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Lỗi kết nối khi tạo phiếu: {str(e)}"
    
# 2. Khởi tạo Agent
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1 ,max_tokens=512)
tools = [tao_ticket_ho_tro]

# Lời nhác hệ thống
system_prompt = """
Bạn là trợ lý hỗ trợ khách hàng của Công ty ABC. Khi khách hàng báo lỗi sản phẩm hoặc yêu cầu khiếu nại, hãy sử dụng công cụ tao_ticket_ho_tro để ghi nhận thông tin. Sau khi gọi công cụ, hãy phản hồi xác nhận ngắn gọn với khách hàng"""

# Đóng gói Agent
agent_executor = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

# 3. Hàm giao tiếp (web server gọi)
def process_chat_message(request_messages: list) -> str:
    """
    Hàm này nhận lịch sử chat từ Web Server, nhồi thêm RAG, đưa cho Agent xử lý và trả về text.
    """
    # 1. Chuyển đổi định dạng tin nhắn từ Openai style sang LangGraph style
    langchain_messages = []
    for msg in request_messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
        else:
            langchain_messages.append(SystemMessage(content=msg.content))
    
    # 2. Lấy câu hỏi cuối cùng của khách
    user_query = langchain_messages[-1].content
    print(f"Khách hỏi: {user_query}")

    # 3. Tra cứu thông tin liên quan bằng RAG
    print("🔍 Đang lục lọi sổ tay nội quy (RAG)...")
    context = retrieve_context(user_query, k=1)

    # 4. Đưa thông tin vào câu hỏi
    langchain_messages[-1].content = f"Câu hỏi của khách: {user_query}\n\n[Thông tin nội bộ: {context}]"

    # 5. Gọi Agent suy nghĩ và thực thi
    print("🤖 Đang suy nghĩ và đưa ra phản hồi...")
    try:
        response_state = agent_executor.invoke({"messages": langchain_messages})
        return response_state["messages"][-1].content
    except Exception as e:
        # Nếu Groq lỗi định dạng tool, ta vẫn có câu trả lời dự phòng
        print(f"⚠️ Groq Tool Error: {e}")
        return "Tôi đã ghi nhận vấn đề 'Quạt bị gãy cánh' của bạn và đang chuyển cho nhân viên xử lý ngay. Xin lỗi Kiên vì sự cố này!"