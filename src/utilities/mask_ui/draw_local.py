import numpy as np
import io

from PIL import Image
from flask import request, jsonify

def draw_mask_local():
    data = request.get_json(silent=True)

    if not data or "mask" not in data:
        return jsonify({"error": "Mask data is missing"}), 400

    try:
        mask_b64 = data["mask"]

        if "," in mask_b64:
            mask_b64 = mask_b64.split(",", 1)[1]

        import base64

        binary = base64.b64decode(mask_b64)

        with Image.open(io.BytesIO(binary)) as img:
            mask = np.array(img.convert("L"))

        mask = (mask > 127).astype(np.uint8) * 255

        return mask

    except Exception as e:
        return jsonify({"error": str(e)}), 400