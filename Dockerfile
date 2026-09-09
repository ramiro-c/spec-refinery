FROM python:3.12-slim

WORKDIR /app

COPY requirements-compose.txt .
RUN pip install --no-cache-dir -r requirements-compose.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
