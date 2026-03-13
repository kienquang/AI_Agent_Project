import streamlit as st
import requests
import uuid

# Cấu hình giao diện
st.set_page_config(page_title="AI Agent Support", page_icon="🤖")
st.title("🤖 AI Support Agent (Human-in-the-Loop)")

# Khởi tạo Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Biến để khóa khung chat khi đang chờ xác nhận
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# HÀM GỬI REQUEST LÊN BACKEND
def send_to_backend(query: str, action: str = "chat"):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": query}],
        "user": st.session_state.session_id,
        "action": action
    }
    
    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
        response_data = response.json()
        
        reply_content = response_data["choices"][0]["message"]["content"]
        requires_conf = response_data["choices"][0]["message"].get("requires_confirmation", False)
        
        # Lưu câu trả lời của AI vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": reply_content})
        st.session_state.awaiting_confirmation = requires_conf
        
        st.rerun() # Tải lại trang để vẽ giao diện mới
    except Exception as e:
        st.error(f"Lỗi kết nối Backend: {e}")

# XỬ LÝ KHUNG CHAT HOẶC NÚT BẤM
if st.session_state.awaiting_confirmation:
    # Nếu đang chờ, hiển thị 2 Nút bấm to đùng
    st.warning("⚠️ Hệ thống đang chờ bạn xác nhận lệnh tạo phiếu!")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ ĐỒNG Ý TẠO PHIẾU", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "(Bạn đã bấm ĐỒNG Ý)"})
            send_to_backend("", action="approve")
    with col2:
        if st.button("❌ HỦY BỎ LỆNH NÀY", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "(Bạn đã bấm HỦY BỎ)"})
            send_to_backend("", action="reject")
else:
    # Nếu bình thường, hiển thị khung chat
    if prompt := st.chat_input("Nhập vấn đề bạn đang gặp phải..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Đang suy nghĩ..."):
            send_to_backend(prompt, action="chat")