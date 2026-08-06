FROM python:3.10-slim

WORKDIR /env

COPY requirements.txt .

# Installazione dipendenze di sistema e Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
