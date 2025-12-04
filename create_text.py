
import os

output_dir = "dummy_dataset/text"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "sample1.txt"), "w") as f:
    f.write("This is a dummy text file.")
