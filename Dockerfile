FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV AISF_JWT_SECRET=change-me
EXPOSE 5000
CMD ["python","app.py"]
