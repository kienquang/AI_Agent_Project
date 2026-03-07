import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)

COLLECTION_NAME = "chinh_sach_cong_ty"

# Khởi tạo biến embeddings và vectorstore ở biến Global để không phải load lại
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
    use_jsonb=True,
)

def retrieve_context(query: str, top_k: int = 2) -> str:
    """Hàm RAG: Truy vấn PGVector để lấy ngữ cảnh liên quan nhất"""
    try:
        # gọi lệnh similarity search để tìm kiếm vector trong db
        docs = vectorstore.similarity_search(query, k=top_k)

        # Gộp các đoạn văn bản tìm được
        context = "\n---\n".join([doc.page_content for doc in docs])
        return context
    
    except Exception as e:
        print(f"❌ [RAG Engine] Lỗi khi tra cứu DB: {e}")
        return "Lỗi hệ thống tra cứu tài liệu nội bộ."
    
# Test nhanh nếu chạy file này trực tiếp
if __name__ == "__main__":
    print("Test tra cứu câu hỏi: 'Bảo hành điện thoại mấy tháng?'")
    ket_qua = retrieve_context("yêu cầu tuyển thực tập sinh")
    print(f"\nKết quả:\n{ket_qua}")