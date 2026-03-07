import os
from dotenv import load_dotenv

# Import các công cụ xử lý văn bản
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Import công cụ Vector Database của Postgres
from langchain_postgres import PGVector

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

# Thư viện SQLAlchemy (dùng cho PGVector) yêu cầu giao thức bắt đầu 
# bằng "postgresql+psycopg://" thay vì "postgresql://"
if DB_URL and DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)

DATA_PATH = "data/chinh_sach_abc.md"
COLLECTION_NAME = "chinh_sach_cong_ty"

def upload_to_pgvector():
    if not DB_URL:
        raise ValueError("Chưa cấu hình DATABASE_URL trong file .env!")
    
    print("⚙️ Đang tải mô hình ngôn ngữ (Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"📖 Đang đọc file tài liệu từ {DATA_PATH}...")
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()

    print("✂️ Đang băm nhỏ tài liệu (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"   -> Đã cắt thành {len(chunks)} đoạn nhỏ.")

    print("☁️ Đang kết nối tới PostgreSQL (Neon.tech) và tải dữ liệu lên...")
    # Khởi tạo kết nối Vector Database
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True, # Tối ưu hóa lưu trữ JSON trên Postgres
    )

    # xóa dữ liệu cũ để tránh trùng lặp khi chạy lại file này
    vectorstore.drop_tables()

    #  TỰ TẠO BẢNG
    vectorstore.create_tables_if_not_exists()
    vectorstore.create_collection()

    # Bơm dữ liệu mới vào
    vectorstore.add_documents(chunks)
    print("✅ Đã tải dữ liệu lên PGVector thành công!")

if __name__ == "__main__":
    upload_to_pgvector()