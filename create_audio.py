
import os
import numpy as np
import soundfile as sf

output_dir = "dummy_dataset/audio"
os.makedirs(output_dir, exist_ok=True)

samplerate = 44100
duration = 1
frequency = 440

t = np.linspace(0., duration, int(samplerate * duration))
amplitude = np.iinfo(np.int16).max * 0.5
data = amplitude * np.sin(2. * np.pi * frequency * t)

sf.write(os.path.join(output_dir, 'sample1.wav'), data.astype(np.int16), samplerate)
