FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y libgomp1 
    
RUN apt-get update && \
apt-get install -y \
ffmpeg \
libsm6 \
libxext6 \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "demo.py"]