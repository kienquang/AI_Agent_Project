# Sử dụng Python 3.11 làm nền móng
FROM python:3.11-slim

# Thiết lập thư mục làm việc bên trong Container
WORKDIR /app

# Copy file requirements.txt vào và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn của dự án (core, data, main.py...) vào Container
COPY . .

# Mở cổng 8000 để giao tiếp
EXPOSE 8000

# Lệnh để khởi động Server khi Container chạy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]