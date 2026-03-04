FROM python:3.11-slim

# Tạo một user mới để bảo mật (HF yêu cầu không chạy quyền root)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

# HF mặc định chạy cổng 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]