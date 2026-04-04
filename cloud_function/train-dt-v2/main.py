# Train on all data < today (local TZ); hold out today
# HTTP entrypoint: train_dt_http

import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    client.bucket(bucket).blob(key).upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _write_json_to_gcs(client: storage.Client, bucket: str, key: str, payload: dict):
    client.bucket(bucket).blob(key).upload_from_string(
        json.dumps(payload, indent=2), content_type="application/json"
    )


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", str(name)).strip("-").lower()


def _calc_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None
    bias = float(np.mean(y_pred - y_true))
    return {"mae": mae, "rmse": rmse, "mape": mape, "bias": bias}


def _manual_pdp(estimator, X_ref: pd.DataFrame, feature: str, max_grid_points: int = 20):
    x = X_ref[feature]
    if pd.api.types.is_numeric_dtype(x):
        x_nonnull = x.dropna()
        if x_nonnull.empty:
            return pd.DataFrame()
        grid = np.unique(np.quantile(x_nonnull, np.linspace(0.05, 0.95, max_grid_points)))
    else:
        grid = x.astype(str).fillna("MISSING").value_counts().head(max_grid_points).index.tolist()

    rows = []
    for g in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature] = g
        rows.append({"feature": feature, "grid_value": g, "pdp_mean": float(np.mean(estimator.predict(X_tmp)))})
    return pd.DataFrame(rows)


def run_once(dry_run: bool = False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    if missing := (required - set(df.columns)):
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Time split ---
    df["scraped_at_dt_utc"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]

    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean ---
    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    unique_dates = sorted(df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two dates"}

    today = unique_dates[-1]
    train_df = df[df["date_local"] < today].dropna(subset=["price_num"])
    holdout_df = df[df["date_local"] == today].dropna(subset=["price_num"])

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few rows"}

    # --- Features (FIXED HERE) ---
    cat_cols = [c for c in ["make", "model", "type", "fuel", "paint_color", "city", "state", "zip_code"] if c in df.columns]
    num_cols = [c for c in ["year_num", "mileage_num"] if c in df.columns]
    feats = cat_cols + num_cols

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("model", DecisionTreeRegressor(random_state=42))
    ])

    grid = GridSearchCV(
        pipe,
        {
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [2, 5, 10],
        },
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1
    )

    grid.fit(train_df[feats], train_df["price_num"])
    model = grid.best_estimator_

    # --- Predict ---
    preds_df = pd.DataFrame()
    if not holdout_df.empty:
        y_hat = model.predict(holdout_df[feats])
        preds_df = holdout_df.copy()
        preds_df["pred_price"] = y_hat

    # --- Save ---
    now = pd.Timestamp.utcnow().strftime("%Y%m%d%H")
    base = f"{OUTPUT_PREFIX}/{now}"

    if not dry_run:
        if not preds_df.empty:
            _write_csv_to_gcs(client, GCS_BUCKET, f"{base}/preds.csv", preds_df)

    return {"status": "ok", "rows": len(preds_df)}


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=body.get("dry_run", False))
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"})
