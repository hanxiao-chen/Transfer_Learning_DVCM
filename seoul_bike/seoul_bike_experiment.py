"""UCI Seoul Bike Sharing Demand experiment.

Y = log1p(Rented Bike Count)
U = Hour, normalized to [0, 1]
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uci_experiment_common import evaluate_linear_methods_cv_bandwidth, zscore_safe


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
OUTCOME_LABEL = "log_rentals_hour"
DOMAIN_LABEL = "hour"
PREFIX = f"seoul_bike_{OUTCOME_LABEL}"
BANDWIDTH_GRID = [0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def download_if_needed() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "SeoulBikeData.csv"
    if csv_path.exists():
        return csv_path
    zip_path = DATA_DIR / "seoul_bike_sharing_demand.zip"
    if not zip_path.exists():
        from urllib.request import urlretrieve

        logging.info("Downloading %s", URL)
        urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found after extracting {zip_path}")
    return csv_path


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        key = key.replace("°", "")
        key = key.replace("(", "").replace(")", "").replace("/", " ")
        key = "_".join(key.split())
        key = key.replace("%", "")
        rename[col] = key
    return df.rename(columns=rename)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    path = download_if_needed()
    logging.info("Reading %s", path)
    df = pd.read_csv(path, encoding="unicode_escape")
    df = clean_columns(df)
    if "functioning_day" in df.columns:
        df = df[df["functioning_day"].astype(str).str.lower().eq("yes")].copy()

    required = [
        "rented_bike_count",
        "hour",
        "temperaturec",
        "humidity",
        "wind_speed_m_s",
        "visibility_10m",
        "dew_point_temperaturec",
        "solar_radiation_mj_m2",
        "rainfallmm",
        "snowfall_cm",
        "seasons",
        "holiday",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Seoul Bike file is missing required columns: {missing}; columns={df.columns.tolist()}")

    model = pd.DataFrame(index=df.index)
    model["Y"] = np.log1p(df["rented_bike_count"].astype(float))
    model[DOMAIN_LABEL] = df["hour"].astype(float)
    model["U"] = model[DOMAIN_LABEL] / 23.0

    base_features = [
        "temperaturec",
        "humidity",
        "wind_speed_m_s",
        "visibility_10m",
        "dew_point_temperaturec",
        "solar_radiation_mj_m2",
        "rainfallmm",
        "snowfall_cm",
    ]
    for col in base_features:
        model[f"{col}_z"] = zscore_safe(df[col])

    dummy_cols = []
    for col in ["seasons", "holiday"]:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        if dummies.shape[1] > 1:
            dummies = dummies.iloc[:, 1:]
        model = pd.concat([model, dummies.astype(float)], axis=1)
        dummy_cols.extend(dummies.columns.tolist())

    hour_dummies = pd.get_dummies(model[DOMAIN_LABEL].astype(int), prefix="hour", dummy_na=False)
    if hour_dummies.shape[1] > 1:
        hour_dummies = hour_dummies.iloc[:, 1:]
    model = pd.concat([model, hour_dummies.astype(float)], axis=1)

    feature_cols = [f"{col}_z" for col in base_features] + dummy_cols
    pooled_feature_cols = feature_cols + hour_dummies.columns.tolist()
    model = model.dropna(subset=["Y", "U", DOMAIN_LABEL, *pooled_feature_cols]).copy()
    return model, feature_cols, pooled_feature_cols


def main() -> None:
    setup_logging()
    df_model, feature_cols, pooled_feature_cols = build_dataset()
    df_model.to_parquet(DATA_DIR / f"processed_{PREFIX}.parquet", index=False)
    logging.info("Processed Seoul Bike data: n=%d, features=%d", len(df_model), len(feature_cols))

    logging.info("START target-domain bandwidth CV over %s", BANDWIDTH_GRID)
    result, rep_result = evaluate_linear_methods_cv_bandwidth(
        df_model,
        feature_cols,
        domain_label=DOMAIN_LABEL,
        bandwidth_grid=BANDWIDTH_GRID,
        pooled_feature_cols=pooled_feature_cols,
        min_target_size=30,
        bias_order=2,
    )
    logging.info("FINISH bandwidth CV; TL MSE=%.6f", rep_result["TL"].mean())

    result.to_csv(DATA_DIR / f"{PREFIX}_cv_bandwidth_result.csv", index=False)
    rep_result.to_csv(DATA_DIR / f"{PREFIX}_cv_bandwidth_rep_level.csv", index=False)

    bandwidth_summary = (
        rep_result.groupby("selected_bandwidth", as_index=False)
        .agg(n_selected=("selected_bandwidth", "size"), TL_mse_mean=("TL", "mean"))
        .sort_values("selected_bandwidth")
    )
    bandwidth_summary.to_csv(DATA_DIR / f"{PREFIX}_cv_bandwidth_selection_summary.csv", index=False)

    print("\nSelected bandwidth summary:")
    print(bandwidth_summary.to_string(index=False))
    print("\nCV result:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
