FROM python:3.7.9

RUN mkdir -p /usr/src/app/
WORKDIR /usr/src/app/

COPY . /usr/src/app/

RUN pip install --upgrade pip
RUN pip install -r /usr/src/app/requirements.txt
RUN pip install /usr/src/app/packages/cnn_model

EXPOSE 5000

CMD ["python", "./packages/web_api/run.py"]