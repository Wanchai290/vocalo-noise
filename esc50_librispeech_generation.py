import librosa
import soundfile
import numpy as np
import os
from constants import OUTPUT_DIR, MAX_NOISE_FILES, GENERATED_INPUT_DIRECTORY
from util import normalize

ESC50_DATASET_DIR = os.path.join("ESC-50-master", "audio")

def generate():
    noise_files = os.listdir(ESC50_DATASET_DIR)
    noise_files = noise_files[:MAX_NOISE_FILES] # limiting to not have a gazillion data generated
    # pre-load input files
    noise_waveforms = [normalize(librosa.load(os.path.join(ESC50_DATASET_DIR, nf))[0]) for nf in noise_files]
    
    speech_files = os.listdir(OUTPUT_DIR)
    for out_f in speech_files:
        out_waveform, _ = librosa.load(os.path.join(OUTPUT_DIR, out_f))
        
        for i, noise in enumerate(noise_waveforms):
            # find out how much you have to lengthen the input noise
            scaler = int(np.ceil(1.0 * len(out_waveform) / len(noise)))
            noise = noise.tolist() * scaler
            noise = noise[:len(out_waveform)]
            fused = out_waveform + noise

            soundfile.write(os.path.join(GENERATED_INPUT_DIRECTORY, out_f.replace('wav', f'{i}.wav')), fused, 22050)



if __name__ == '__main__':
    try:
        os.mkdir(GENERATED_INPUT_DIRECTORY)
    except FileExistsError:
        pass
    generate()