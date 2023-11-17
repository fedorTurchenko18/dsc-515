# client/Dockerfile
FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install flwr tensorflow numpy

ENTRYPOINT ["python", "TFClient.py"]
