FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prisma ./prisma
RUN prisma generate

COPY app ./app
COPY README.md ./
COPY .env.example ./

ENV PORT=8003
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
