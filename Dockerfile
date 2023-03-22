FROM python:3.10
RUN  python3 -m pip install --upgrade pip

WORKDIR /FLASK-DOCKER

COPY . .

RUN  python3 -m pip install -r requirements.txt


CMD ["python","app.py"]