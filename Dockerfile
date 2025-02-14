FROM python:3.12.6-slim

RUN apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["make", "format"]
