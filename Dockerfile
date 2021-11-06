FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Tabio Setup
RUN apt-get -y update && \
    apt-get install libglib2.0-dev libgirepository1.0-dev libcairo2-dev poppler-utils default-jre-headless -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Tabio requirements
COPY requirements.txt . 
RUN python3 -m pip install --upgrade --no-cache-dir pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Tabio app
COPY . . 
