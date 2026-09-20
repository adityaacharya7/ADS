from PIL import Image
import pathlib

for f in sorted(pathlib.Path("Screenshots").glob("*.png")):
    im = Image.open(f)
    print(f"{f.name}: size={im.size}, mode={im.mode}")
