FROM python:3.12-slim

WORKDIR /app

# 系統相依（PyMuPDF 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY app ./app
COPY sample_docs ./sample_docs
# v1.1 Optimizer 引擎 + demo seeder（供可選的一次性 optimizer-cli profile 使用；
# 不新增常駐 worker。API 服務預設不執行任何 optimizer search）。
COPY evaluation ./evaluation
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
