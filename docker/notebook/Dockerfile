FROM python:3.7

EXPOSE 8888

WORKDIR /usr/src/app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY . .