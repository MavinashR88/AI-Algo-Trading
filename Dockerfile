FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download VADER sentiment lexicon
RUN python -c "import nltk; nltk.download('vader_lexicon', download_dir='/usr/local/nltk_data')"

COPY . .

RUN mkdir -p /tmp && chmod 777 /tmp

EXPOSE 7860

ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DB_PATH=/tmp/trading.db \
    NLTK_DATA=/usr/local/nltk_data

CMD ["python", "app.py"]
