FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY ./src/main.py /app/main.py