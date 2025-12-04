
import os
from PIL import Image

output_dir = "dummy_dataset/image"
os.makedirs(output_dir, exist_ok=True)

img = Image.new('RGB', (60, 30), color = 'black')
img.save(os.path.join(output_dir, 'sample1.png'))
