
import numpy as np
import soundfile as sf
import os

os.makedirs("dummy_validation_dataset/audio", exist_ok=True)
samplerate = 44100
duration = 1
frequency = 440

t = np.linspace(0., duration, int(samplerate * duration))
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)

sf.write('dummy_validation_dataset/audio/sample1.wav', data.astype(np.int16), samplerate)
