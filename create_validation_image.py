
from PIL import Image
import os

os.makedirs("dummy_validation_dataset/image", exist_ok=True)
img = Image.new('RGB', (60, 30), color = 'black')
img.save('dummy_validation_dataset/image/sample1.png')
