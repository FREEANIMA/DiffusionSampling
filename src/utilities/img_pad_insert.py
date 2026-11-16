import cv2
import numpy as np

class ImageDecodeError(Exception):
    pass

def img_pad(img_bytes, patch_size, compression_ratio):
    unit = patch_size * compression_ratio

    try:
        arr = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
    except (TypeError, ValueError, cv2.error) as e:
        raise ImageDecodeError(f"{type(e).__name__}: {e}") from e

    if arr is None:
        raise ImageDecodeError(
            f"size={len(img_bytes)}, head={img_bytes[:16]!r}"
        )

    H, W = arr.shape[:2]

    # 다음 unit 배수로 올림
    new_H = max(unit, ((H + unit - 1) // unit) * unit)
    new_W = max(unit, ((W + unit - 1) // unit) * unit)

    pad_h = new_H - H
    pad_w = new_W - W

    # 중앙 패딩
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    arr = cv2.copyMakeBorder(
        arr,
        top, bottom, left, right,
        cv2.BORDER_REFLECT_101
    )

    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(arr)

    return arr, new_H, new_W, (left, top, right, bottom)