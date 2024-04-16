from flask import Flask, request, jsonify
import io
import pydub
import numpy as np
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

app = Flask(__name__)
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

def process(uploaded_file):
    
    try:
        audio_bytes = uploaded_file.read()
        audio_io = io.BytesIO(audio_bytes)

        audio = pydub.AudioSegment.from_file(audio_io)
        channel_sounds = audio.split_to_mono()
        samples = [s.get_array_of_samples() for s in channel_sounds]

        fp_arr = np.array(samples).T.astype(np.float32)
        fp_arr /= np.iinfo(samples[0].typecode).max

        audio, orig_freq = torchaudio.load(io.BytesIO(fp_arr.tobytes()))
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)

        audio_inputs = processor(audios=audio, return_tensors="pt")
        audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="vie")[0].cpu().numpy().squeeze()

        return audio_array_from_audio
    except Exception as e:
        return f"Error occurred: {str(e)}"



@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    text = process(file)

    return jsonify({'text': text})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)