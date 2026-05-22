from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "DailyDelhiClimateTrain.csv"
TEST_PATH = DATA_DIR / "DailyDelhiClimateTest.csv"

OUTPUT_DIR = BASE_DIR / "saved_models"
PLOTS_DIR = BASE_DIR / "plots"
REPORT_DIR = BASE_DIR / "outputs"

TARGET_COL = "meantemp"

MULTIVARIATE_COLS = [
    "meantemp",
    "humidity",
    "wind_speed",
    "meanpressure"
]

TIME_STEPS = 7

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)