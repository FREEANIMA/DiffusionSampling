import base64
import io
from PIL import Image

def b64_to_image(b64: str, mode: str = "RGB") -> Image.Image:
    if "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img.convert(mode)