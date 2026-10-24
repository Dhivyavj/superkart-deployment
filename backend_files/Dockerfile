FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:superkart_api"]
