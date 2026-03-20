from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = EXTRACTED_DIR / "train"
TEST_DIR = EXTRACTED_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space"
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_PATH = MODELS_DIR / "best_model.keras"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"