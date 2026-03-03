import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Đường dẫn tĩnh
DATA_PATH = "data/chinh_sach_abc.md"
FAISS_DB_PATH = "vector_db/faiss_index"

# sử dụng mô hình emmbeding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_or_load_vector_db():
    """
    Hàm này kiểm tra xem Database đã tồn tại trên ổ cứng chưa.
    - Nếu có rồi: Tải lên siêu nhanh.
    - Nếu chưa có: Đọc file .md, cắt nhỏ, mã hóa và lưu xuống ổ cứng.
    """
    if os.path.exists(FAISS_DB_PATH):
        print("📁 Đang tải Vector Database từ ổ cứng...")
        vector_db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_db
    
    print("⚙️ Chưa có Database. Đang tiến hành đọc file và xây dựng RAG...")

    # 1. Đọc dữ liệu từ file thật
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ Đã cắt tài liệu thành {len(chunks)} đoạn nhỏ.")

    # 3. Tạo vector db và lưu cuống ổ cứng
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_DB_PATH)
    print("✅ Đã lưu Vector Database xuống ổ cứng thành công!")

    return vector_db

def retrieve_context(query: str, k: int = 2) -> str:
    """
    Hàm độc lập chuyên làm nhiệm vụ tra cứu thông tin"""
    db = build_or_load_vector_db()
    docs = db.similarity_search(query, k=k)
    context = "\n---\n".join([doc.page_content for doc in docs])
    return context

# Test thử module nếu chạy file này trực tiếp
if __name__ == "__main__":
    test_query = "Thời gian bảo hành phụ kiện là bao lâu?"
    print(f"Câu hỏi: {test_query}")
    print("Kết quả tìm kiếm")
    print(retrieve_context(test_query))
