from flask import Flask, request, jsonify
import io
import pydub
import numpy as np
import torchaudio
# from transformers import AutoProcessor, SeamlessM4Tv2Model
from transformers import pipeline

app = Flask(__name__)

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")

def process():
    audio_path = "/storage/test.wav"
    audio, orig_sr = torchaudio.load(audio_path)

    resampler = torchaudio.transforms.Resample(orig_sr, 16000)
    audio_resampled = resampler(audio)

    audio_array = audio_resampled.numpy().squeeze()

    output = transcriber(audio_array)['text']

    return output


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    text = process()

    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5002)