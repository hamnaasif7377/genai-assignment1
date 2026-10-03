"""
Task 1: Universal Restoration endpoint. Accepts an uploaded image,
runs it through the trained universal autoencoder, and returns the
restored image plus inference timing information.
"""
import time

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
import io

from app.image_utils import preprocess_image, postprocess_image, image_to_bytes
from app.onnx_loader import get_session

router = APIRouter()

MODEL_FILENAME = "task1_universal_autoencoder.onnx"


@router.post("/universal-restoration")
async def universal_restoration(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_array = preprocess_image(image_bytes)

    session = get_session(MODEL_FILENAME)
    input_name = session.get_inputs()[0].name

    start_time = time.perf_counter()
    output = session.run(None, {input_name: input_array})[0]
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    result_img = postprocess_image(output)
    result_bytes = image_to_bytes(result_img)

    return StreamingResponse(
        io.BytesIO(result_bytes),
        media_type="image/png",
        headers={"X-Inference-Time-Ms": f"{inference_time_ms:.2f}"},
    )
