QuickSTT
=====

This time, I provide quick code with streamlit for asr demo, you can run:

```

conda create -n asr python=3.9

conda activate asr

pip install -r requirements.txt

pip install git+https://github.com/huggingface/transformers.git sentencepiece

streamlit run demo/app.py

```