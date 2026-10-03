"""
Shared image preprocessing/postprocessing for ONNX model inference.
Mirrors the exact preprocessing used during training (resize to 128x128,
convert to RGB, scale to [0,1], CHW tensor layout).
"""
import io

import numpy as np
from PIL import Image

IMG_SIZE = 128


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Bytes -> normalized (1, 3, 128, 128) float32 array, matching training preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                      # CHW
    arr = np.expand_dims(arr, axis=0)                 # add batch dim -> (1,3,128,128)
    return arr


def postprocess_image(output_array: np.ndarray) -> Image.Image:
    """Model output (1, 3, 128, 128) float32 in [0,1] -> a displayable PIL Image."""
    arr = output_array[0]                 # drop batch dim -> (3, 128, 128)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)          # CHW -> HWC
    return Image.fromarray(arr)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()
