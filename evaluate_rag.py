import os
import pandas as pd
from dotenv import load_dotenv

# Thư viện dữ liệu
from datasets import Dataset

# Thư viện Ragas
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Độ trung thực 
    answer_relevancy,    # Độ bám sát câu hỏi
    context_precision,   # Độ chính xác của tài liệu lấy ra
    context_recall       # Độ đầy đủ của tài liệu lấy ra so với câu trả lời mẫu
)

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage

# Import RAG Engine 
from core.rag_engine import retrieve_context

load_dotenv()

# 1. Khởi tạo bộ đánh giá
evaluator_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#  2. Tạo dữ liệu kiểm thử
# thực tế có thể lưu file excel/cvs nhưng giờ đơn giản 2 câu
# Trong thực tế, tập này có thể lưu ở file Excel/CSV gồm hàng trăm câu.
# Ở đây ta làm mẫu 2 câu hỏi liên quan đến chính sách công ty.
questions = [
    "Công ty có chính sách đổi trả sản phẩm như thế nào?",
    "Thời gian bảo hành của điện thoại iPhone là bao lâu?"
]

# Câu trả lời kỳ vọng (Ground Truth) do con người viết chuẩn xác
ground_truths = [
    "Khách hàng được đổi trả sản phẩm trong vòng 7 ngày nếu có lỗi từ nhà sản xuất.",
    "Điện thoại iPhone được bảo hành chính hãng 12 tháng."
]

def run_evaluation():
    print("🚀 Bắt đầu quá trình Evaluation (Chấm điểm RAG)...")
    
    answers = []
    contexts = []

    # 3. Chạy thử
    for query in questions:
        print(f"👉 Đang xử lý câu hỏi: '{query}'")

        # Lấy liệu từ PostgreSQL
        retrieved_docs_text = retrieve_context(query)
        contexts.append([retrieved_docs_text])

        # Gọi AI để sinh câu trả lời dựa trne econtext
        prompt = f"""
            Dựa vào TÀI LIỆU sau đây, hãy trả lời câu hỏi: '{query}'

            QUY TẮC LẬP LUẬN BẮT BUỘC (Để đảm bảo tính trung thực):
            Nếu câu hỏi nhắc đến một sản phẩm cụ thể mà tài liệu chỉ quy định chung cho một nhóm, BẠN PHẢI giải thích rõ sự liên quan trong câu trả lời.
            Ví dụ cách trả lời đúng: "Vì điện thoại iPhone thuộc nhóm thiết bị điện tử, nên theo quy định nó sẽ được bảo hành..."

            TÀI LIỆU: {retrieved_docs_text}
            """
        response = evaluator_llm.invoke([SystemMessage(content=prompt)]).content
        answers.append(response)

    # 4. Tạo dữ liệu khóa
    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": ground_truths
    })

    print("\n⚖️ Đang nhờ AI Giám khảo chấm điểm (Quá trình này mất khoảng 1-2 phút)...")

    # Ragas sẽ tạo ra các Prompt đặc biệt để chấm điểm từng tiêu chí
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 5. In báo cáo
    print("\n🏆 KẾT QUẢ ĐÁNH GIÁ CHI TIẾT:")
    df = result.to_pandas()

    # Cách an toàn: In toàn bộ DataFrame để xem Ragas thực sự trả về những gì
    # (Pandas sẽ tự động ẩn bớt cột nếu quá dài)
    print(df.to_string())
    
    # Hoặc nếu bạn chỉ muốn in các cột điểm số mà không sợ lỗi KeyError:
    # Lấy danh sách các cột thực tế đang có trong df
    existing_columns = df.columns.tolist()
    print(f"\nCác cột tìm thấy: {existing_columns}") # Để bạn debug nếu cần
    
    # result.scores là một dictionary chứa điểm trung bình của từng metric
    print("\n📊 ĐIỂM TRUNG BÌNH TOÀN HỆ THỐNG:")
    
    # Dùng df['tên_cột'].mean() của Pandas để tính trung bình cộng cực kỳ an toàn
    print(f"- Context Precision: {df['context_precision'].mean():.2f}/1.0")
    print(f"- Context Recall:    {df['context_recall'].mean():.2f}/1.0")
    print(f"- Faithfulness:      {df['faithfulness'].mean():.2f}/1.0")
    print(f"- Answer Relevancy:  {df['answer_relevancy'].mean():.2f}/1.0")
    print("="*50)
if __name__ == "__main__":
    run_evaluation()