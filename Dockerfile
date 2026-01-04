FROM python:3.10.1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 9000
CMD ["python","training/train_SCT.py","--run-id","test"]