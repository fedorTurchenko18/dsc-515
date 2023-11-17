# server/Dockerfile
FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install flwr tensorflow numpy

CMD ["python", "server.py"]
