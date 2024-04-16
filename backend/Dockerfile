FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y libgomp1 


RUN apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN apt-get install libgomp1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5002
CMD ["python", "api.py"]