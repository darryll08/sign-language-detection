from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import io

from src.landmark_features import HandLandmarkFeatureExtractor, ensure_hand_landmarker_model
from src.landmark_inference import (
    load_landmark_model,
    load_landmark_label_encoder,
    predict_landmark_pil,
)

app = FastAPI()

# Pastikan model hand landmarker tersedia
ensure_hand_landmarker_model()

# Load landmark classifier
landmark_model = load_landmark_model()
label_encoder = load_landmark_label_encoder()
extractor = HandLandmarkFeatureExtractor()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.on_event("shutdown")
def shutdown_event():
    extractor.close()


def read_upload_as_pil(contents: bytes) -> Image.Image:
    with Image.open(io.BytesIO(contents)) as image:
        return image.convert("RGB")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-capture")
async def predict_capture(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_upload_as_pil(contents)

        result = predict_landmark_pil(
            model=landmark_model,
            label_encoder=label_encoder,
            extractor=extractor,
            image=image,
            top_k=3,
        )

        return JSONResponse({
            "ok": True,
            "label": result["label"],
            "confidence": float(result["confidence"]),
            "top3": result["topk"],
            "used_landmarks": bool(result["used_landmarks"]),
            "message": "ok"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Capture prediction failed: {str(e)}")


@app.post("/predict-live")
async def predict_live(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_upload_as_pil(contents)

        result = predict_landmark_pil(
            model=landmark_model,
            label_encoder=label_encoder,
            extractor=extractor,
            image=image,
            top_k=3,
        )

        return JSONResponse({
            "ok": True,
            "label": result["label"],
            "confidence": float(result["confidence"]),
            "top3": result["topk"],
            "used_landmarks": bool(result["used_landmarks"]),
            "message": "ok"
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Live prediction failed: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "landmark_mlp"}