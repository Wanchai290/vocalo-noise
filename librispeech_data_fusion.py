import librosa
import soundfile
import numpy as np
import os
from constants import OUTPUT_DIR
"""For each entry, generates a fused mp3"""

# Librispeech directory format
# A / [B1, .., Bn] / 'A-B-xxxx.flac'
DATASET_PATH = os.path.join("LibriSpeech", "dev-clean")

def read_dir(A: str, B: str) -> list[float]:
	directory = os.path.join(DATASET_PATH, A, B)
	files = [fn for fn in os.listdir(directory) if ".txt" not in fn]
	# open all using librosa & fuse them
	fused_waveform = np.array([])
	sample_rate = 0
	for filename in files:
		print(os.path.join(directory, filename))
		waveform, sr = librosa.load(os.path.join(directory, filename))
		fused_waveform = np.append(fused_waveform, waveform)
		sample_rate = sr
	return fused_waveform, sample_rate

def generate(ids: list[int]):
	for A in ids:
		Bs = [s for s in os.listdir(os.path.join(DATASET_PATH, str(A)))]
		for B in Bs:
			waveform, sr = read_dir(str(A), B)
			soundfile.write(os.path.join(OUTPUT_DIR, f"{str(A)}_{B}.wav"), waveform, sr)

if __name__ == '__main__':
	os.mkdir(OUTPUT_DIR)
	# Librispeech data fusion
	generate([84, 174])