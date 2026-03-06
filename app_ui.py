import streamlit as st
import requests
import uuid

# Cấu hình trang web
st.set_page_config(page_title="Hệ thống AI hỗ trợ ABC",page_icon="🤖")
st.title("🤖 Trợ lý ảo Công ty ABC")
st.caption("Giao diện được xây dựng bằng Streamlit - Nhẹ, nhanh, và hoàn toàn miễn phí.")

# Khởi tạo session_id duy nhất cho trình dyuetej này
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Khởi tạo bộ nhớ lưu lịch sử chat trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập tin nhắn mới
if prompt := st.chat_input("Nhập câu hỏi hoặc yêu cầu của bạn..."):
    # Hiển thị tin nhắn mới trên giao diện
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gửi request đến FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Đang xử lý...")
        try:
            # Gọi API nội bộ của bạn 
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}], # Chỉ cần gửi câu hiện tại
                "user": st.session_state.session_id # Gửi session_id để backend biết đây là phiên nào,
            }
            # Thay link này bằng link Render.com khi deploy lên cloud
            # response = requests.post("https://kiendao744-ai-agent-backend.hf.space/v1/chat/completions", json=payload)
            response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)

            if response.status_code == 200:
                ai_reply = response.json()["choices"][0]["message"]["content"]
                message_placeholder.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            else:
                message_placeholder.error(f"Lỗi từ server: {response.status_code}")
        except Exception as e:
            message_placeholder.error(f"Lỗi kết nối: {e}")