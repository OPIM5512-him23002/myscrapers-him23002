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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    client.bucket(bucket).blob(key).upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv"
    )


def _write_png_to_gcs(client: storage.Client, bucket: str, key: str, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    client.bucket(bucket).blob(key).upload_from_string(
        buf.getvalue(),
        content_type="image/png"
    )
    plt.close(fig)


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "-", str(name)).strip("-").lower()


def _calc_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    mask = y_true != 0
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = None

    bias = float(np.mean(y_pred - y_true))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
    }


def _manual_pdp(estimator, X_ref: pd.DataFrame, feature: str, max_grid_points: int = 20) -> pd.DataFrame:
    x = X_ref[feature]

    if pd.api.types.is_numeric_dtype(x):
        x_nonnull = x.dropna()
        if x_nonnull.empty:
            return pd.DataFrame(columns=["feature", "grid_value", "pdp_mean"])

        n_unique = x_nonnull.nunique()
        if n_unique <= max_grid_points:
            grid = np.sort(x_nonnull.unique())
        else:
            grid = np.unique(np.quantile(x_nonnull, np.linspace(0.05, 0.95, max_grid_points)))
    else:
        grid = x.astype(str).fillna("MISSING").value_counts().head(max_grid_points).index.tolist()

    rows = []
    for g in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature] = g
        preds = estimator.predict(X_tmp)
        rows.append({
            "feature": feature,
            "grid_value": g,
            "pdp_mean": float(np.mean(preds))
        })

    return pd.DataFrame(rows)


def _make_pdp_plot(pdp_df: pd.DataFrame, feature: str):
    fig, ax = plt.subplots(figsize=(6, 4))

    x = pdp_df["grid_value"]
    y = pdp_df["pdp_mean"]

    try:
        x_num = pd.to_numeric(x)
        order = np.argsort(x_num)
        ax.plot(x_num.iloc[order], y.iloc[order], marker="o")
        ax.set_xlabel(feature)
    except Exception:
        ax.plot(range(len(x)), y, marker="o")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels([str(v) for v in x], rotation=45, ha="right")
        ax.set_xlabel(feature)

    ax.set_ylabel("Average predicted price")
    ax.set_title(f"PDP: {feature}")
    fig.tight_layout()
    return fig


def run_once(dry_run: bool = False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]

    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct dates",
            "dates": [str(d) for d in unique_dates]
        }

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()].copy()
    holdout_df = holdout_df[holdout_df["price_num"].notna()].copy()

    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info("Train rows after target clean: %d (dropped_for_target=%d)", len(train_df), dropped_for_target)
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Feature setup ---
    target = "price_num"

    candidate_cat_cols = ["make", "model", "type", "fuel", "paint_color", "city", "state", "zip_code"]
    candidate_num_cols = ["year_num", "mileage_num"]

    cat_cols = [c for c in candidate_cat_cols if c in train_df.columns]
    num_cols = [c for c in candidate_num_cols if c in train_df.columns]
    feats = cat_cols + num_cols

    if not feats:
        raise ValueError("No usable features found for training")

    logging.info("Using features: %s", feats)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    X_train = train_df[feats].copy()
    y_train = train_df[target].copy()

    pipe = Pipeline([
        ("pre", pre),
        ("model", DecisionTreeRegressor(random_state=42))
    ])

    param_grid = {
        "model__max_depth": [5, 8, 12, None],
        "model__min_samples_leaf": [2, 5, 10],
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    logging.info("Running GridSearchCV...")
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    logging.info("Best params: %s", grid.best_params_)
    logging.info("Best CV MAE: %.4f", -grid.best_score_)

    # --- Predict today's holdout ---
    preds_df = pd.DataFrame()
    metrics = {"mae": None, "rmse": None, "mape": None, "bias": None}

    if not holdout_df.empty:
        X_h = holdout_df[feats].copy()
        y_true = holdout_df["price_num"].to_numpy()
        y_hat = best_model.predict(X_h)

        base_cols = [c for c in [
            "post_id", "scraped_at", "make", "model", "year", "mileage", "price",
            "type", "fuel", "paint_color", "city", "state", "zip_code"
        ] if c in holdout_df.columns]

        preds_df = holdout_df[base_cols].copy()
        preds_df["actual_price"] = y_true
        preds_df["pred_price"] = np.round(y_hat, 2)
        preds_df["residual"] = np.round(preds_df["pred_price"] - preds_df["actual_price"], 2)

        metrics = _calc_regression_metrics(y_true, y_hat)

    # --- Permutation importance for all features ---
    permimp_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    if not holdout_df.empty and len(holdout_df) >= 5:
        X_h = holdout_df[feats].copy()
        y_true = holdout_df["price_num"].to_numpy()

        perm = permutation_importance(
            best_model,
            X_h,
            y_true,
            n_repeats=10,
            random_state=42,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        permimp_df = pd.DataFrame({
            "feature": X_h.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # --- PDPs for top 3 features ---
    pdp_csv_keys = []
    pdp_plot_keys = []

    top3_features = permimp_df["feature"].head(3).tolist() if not permimp_df.empty else feats[:3]

    pdp_dfs = {}
    for feat in top3_features:
        try:
            pdp_df = _manual_pdp(best_model, X_train, feat, max_grid_points=20)
            if not pdp_df.empty:
                pdp_dfs[feat] = pdp_df
        except Exception as e:
            logging.warning("Skipping PDP for feature '%s': %s", feat, e)

    # --- Output paths ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_hour = now_utc.strftime("%Y%m%d%H")
    out_dir = f"{OUTPUT_PREFIX}/{run_hour}"

    preds_key = f"{out_dir}/preds-llm.csv"
    perm_key = f"{out_dir}/permimp-llm.csv"

    if not dry_run:
        if len(preds_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET, preds_key, preds_df)
            logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, preds_key, len(preds_df))

        if not permimp_df.empty:
            _write_csv_to_gcs(client, GCS_BUCKET, perm_key, permimp_df)
            logging.info("Wrote permutation importance to gs://%s/%s", GCS_BUCKET, perm_key)

        for feat, pdp_df in pdp_dfs.items():
            pdp_csv_key = f"{out_dir}/pdp-llm-{_safe_slug(feat)}.csv"
            _write_csv_to_gcs(client, GCS_BUCKET, pdp_csv_key, pdp_df)
            pdp_csv_keys.append(pdp_csv_key)
            logging.info("Wrote PDP data for %s to gs://%s/%s", feat, GCS_BUCKET, pdp_csv_key)

            fig = _make_pdp_plot(pdp_df, feat)
            pdp_plot_key = f"{out_dir}/pdp-llm-{_safe_slug(feat)}.png"
            _write_png_to_gcs(client, GCS_BUCKET, pdp_plot_key, fig)
            pdp_plot_keys.append(pdp_plot_key)
            logging.info("Wrote PDP plot for %s to gs://%s/%s", feat, GCS_BUCKET, pdp_plot_key)
    else:
        logging.info("Dry run; skipping artifact writes under gs://%s/%s", GCS_BUCKET, out_dir)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "best_params": grid.best_params_,
        "best_cv_mae": float(-grid.best_score_),
        **metrics,
        "features_used": feats,
        "top3_pdp_features": list(pdp_dfs.keys()),
        "predictions_key": preds_key if len(preds_df) > 0 else None,
        "permimp_key": perm_key if not permimp_df.empty else None,
        "pdp_csv_keys": pdp_csv_keys,
        "pdp_plot_keys": pdp_plot_keys,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False))
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
