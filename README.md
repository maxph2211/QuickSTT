QuickSTT
=====

This time, I provide quick code with streamlit for asr demo, you can simply run:

```
docker-compose up -d --build

```

or manually seting up the environment and run the app: 

```

conda create -n asr python=3.9

conda activate asr

pip install -r backend/requirements.txt

python backend/api.py

pip install git+https://github.com/huggingface/transformers.git sentencepiece

streamlit run frontend/demo.py

```