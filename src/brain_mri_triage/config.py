from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
EXTRACTED_DATASET_DIR = DATA_DIR / "mendeley"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRADCAM_DIR = ARTIFACTS_DIR / "grad_cam"
CASE_REPORTS_DIR = ARTIFACTS_DIR / "case_reports"

BEST_MODEL_PATH = ARTIFACTS_DIR / "melhor_modelo_mendeley.keras"
FINAL_MODEL_PATH = ARTIFACTS_DIR / "modelo_mendeley_multiclasse.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"
HISTORY_PATH = ARTIFACTS_DIR / "analysis_history.csv"
VALIDATION_HISTORY_PATH = ARTIFACTS_DIR / "developer_validations.csv"
CASE_IMAGES_DIR = ARTIFACTS_DIR / "case_images"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.csv"
TRAINING_LOG_PATH = ARTIFACTS_DIR / "training_log.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.csv"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

LOW_CONFIDENCE_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.80

TRAIN_DIR_NAMES = {"train", "training"}
TEST_DIR_NAMES = {"test", "testing"}

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Tumor pituitário",
    "pituitario": "Tumor pituitário",
    "pituitarytumor": "Tumor pituitário",
    "notumor": "Sem tumor aparente",
    "notumour": "Sem tumor aparente",
    "no_tumor": "Sem tumor aparente",
    "no tumor": "Sem tumor aparente",
    "sem tumor": "Sem tumor aparente",
    "sem_tumor": "Sem tumor aparente",
}


def normalize_class_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def compact_class_name(name: str) -> str:
    return normalize_class_name(name).replace("_", "")


def display_class_name(name: str) -> str:
    normalized = normalize_class_name(name)
    compact = compact_class_name(name)
    return CLASS_DISPLAY_NAMES.get(normalized) or CLASS_DISPLAY_NAMES.get(compact) or str(name)


def is_no_tumor_class(name: str) -> bool:
    compact = compact_class_name(name)
    return compact in {"notumor", "notumour", "sem_tumor", "semtumor"}


def is_pituitary_class(name: str) -> bool:
    compact = compact_class_name(name)
    return compact in {"pituitary", "pituitarytumor", "pituitario", "tumorpituitario"}
