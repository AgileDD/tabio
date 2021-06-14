FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Env Setup
RUN apt-get update && \
    apt-get install libglib2.0-dev libgirepository1.0-dev libcairo2-dev poppler-utils default-jre-headless -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Tabio
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt && \
    python3 -m pip install python-multipart ujson
COPY . /app

ENTRYPOINT [ "/start-reload.sh" ]
