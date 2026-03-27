"""
main.py — Improved version

Changes vs original:
- Two separate HandLandmarkFeatureExtractor instances:
    * capture_extractor: IMAGE mode (single shot, independent)
    * live_extractor:    VIDEO mode (temporal tracking for spell mode)
- live_smoother: EMA probability smoother attached to /predict-live
  so consecutive frames get smoothed probability vectors — more stable labels
- /predict-live now passes use_video_mode=True + the smoother
- /predict-capture resets the live smoother on each capture
  to avoid stale video-mode state leaking into capture mode
- startup/shutdown properly closes both extractors
"""

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from PIL import Image
import io

from src.landmark_features import HandLandmarkFeatureExtractor, ensure_hand_landmarker_model
from src.landmark_inference import (
    load_landmark_model,
    load_landmark_label_encoder,
    predict_landmark_pil,
    LandmarkSmoother,
)

# ── Global state ─────────────────────────────────────────────────────────────

landmark_model = None
label_encoder = None
capture_extractor: HandLandmarkFeatureExtractor = None   # IMAGE mode
live_extractor: HandLandmarkFeatureExtractor = None      # VIDEO mode
live_smoother: LandmarkSmoother = None

# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global landmark_model, label_encoder
    global capture_extractor, live_extractor, live_smoother

    ensure_hand_landmarker_model()

    landmark_model = load_landmark_model()
    label_encoder = load_landmark_label_encoder()

    # IMAGE mode extractor — used for single-shot /predict-capture
    capture_extractor = HandLandmarkFeatureExtractor(
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.50,
        enable_video_mode=False,
    )

    # VIDEO mode extractor — used for real-time /predict-live
    # Lower detection threshold so fast hand movements are not dropped
    live_extractor = HandLandmarkFeatureExtractor(
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.40,
        enable_video_mode=True,
    )

    # EMA smoother for live mode (alpha=0.45: fairly reactive)
    live_smoother = LandmarkSmoother(alpha=0.45)

    yield  # app is running

    capture_extractor.close()
    live_extractor.close()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


def read_upload_as_pil(contents: bytes) -> Image.Image:
    with Image.open(io.BytesIO(contents)) as image:
        return image.convert("RGB")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-capture")
async def predict_capture(file: UploadFile = File(...)):
    """
    Single-shot prediction for Capture Mode.
    Uses IMAGE mode extractor — no temporal state.
    Also resets the live smoother so previous video state doesn't bleed in.
    """
    try:
        contents = await file.read()
        image = read_upload_as_pil(contents)

        # Reset live smoother so stale EMA doesn't affect next spell session
        live_smoother.reset()

        result = predict_landmark_pil(
            model=landmark_model,
            label_encoder=label_encoder,
            extractor=capture_extractor,
            image=image,
            top_k=3,
            use_video_mode=False,
            smoother=None,  # no smoothing for single captures
        )

        return JSONResponse(
            {
                "ok": True,
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "top3": result["topk"],
                "used_landmarks": bool(result["used_landmarks"]),
                "hand_detected": bool(result["hand_detected"]),
                "message": "ok",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Capture prediction failed: {str(e)}")


@app.post("/predict-live")
async def predict_live(file: UploadFile = File(...)):
    """
    Real-time prediction for Spell Mode.
    Uses VIDEO mode extractor + EMA smoother for temporal stability.
    """
    try:
        contents = await file.read()
        image = read_upload_as_pil(contents)

        result = predict_landmark_pil(
            model=landmark_model,
            label_encoder=label_encoder,
            extractor=live_extractor,
            image=image,
            top_k=3,
            use_video_mode=True,
            smoother=live_smoother,
        )

        return JSONResponse(
            {
                "ok": True,
                "label": result["label"],
                "confidence": float(result["confidence"]),
                "top3": result["topk"],
                "used_landmarks": bool(result["used_landmarks"]),
                "hand_detected": bool(result["hand_detected"]),
                "message": "ok",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Live prediction failed: {str(e)}")


@app.post("/reset-live")
async def reset_live():
    """
    Reset the video-mode temporal state.
    Call this from the frontend when the user stops/starts spell mode,
    so previous hand tracking state is cleared.
    """
    live_smoother.reset()
    # Reset video timestamp so MediaPipe doesn't get confused
    if live_extractor is not None:
        live_extractor._video_timestamp_ms = 0
    return JSONResponse({"ok": True, "message": "live state reset"})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "landmark_mlp",
        "capture_mode": "IMAGE",
        "live_mode": "VIDEO + EMA smoother",
    }
