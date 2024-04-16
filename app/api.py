from flask import Flask, request, jsonify
import io
import pydub
import numpy as np
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

app = Flask(__name__)


def process(file_path):
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    audio, orig_freq = torchaudio.load(file_path)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) 
    audio_inputs = processor(audios=audio, return_tensors="pt")
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
    return audio_array_from_audio


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    audio = pydub.AudioSegment.from_file(file)
    channel_sounds = audio.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    text = process(io.BytesIO(fp_arr[:, 0]))
    
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)
