import os

# =================
# == Path config ==
# =================

OUT_DIR = "../"
RAW_DATA_PATH = "datasets/raw/"
PROCESSED_DATA_PATH = "datasets/processed/"

# =====================
# == Global variable ==
# =====================

TRAIN_LS = [0, 1, 2, 3, 4, 5]
VAL_LS   = [6, 7]
TEST_LS  = [8, 9]

LABEL_MAP = {
    "Antenna": 0,
    "Cable": 1,
    "Electric pole": 2,
    "Wind turbine": 3,
    "Other": 33
}