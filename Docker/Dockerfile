FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
ENV DEBIAN_FRONTEND noninteractive

# Env Setup
RUN apt-get update && \
    apt-get install libglib2.0-dev libgirepository1.0-dev libcairo2-dev poppler-utils default-jre-headless -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Tabio
COPY ./app /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/tabio/requirements.txt && \
    python3 -m pip install python-multipart ujson

ENTRYPOINT [ "/start-reload.sh" ]