import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from app.mlops.mlflow_utils import is_enabled, log_training_run
from app.pipeline.models import MODEL_CANDIDATES, build_model


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    return 0.0 if denom == 0 else float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate(y_true, y_pred):
    return {
        "WAPE": round(wape(y_true, y_pred) * 100, 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def tune_models(X, y, dataset_key: str | None = None):
    """Evaluate forecasting models using time-series CV, refit winner, log to MLflow."""
    started = time.perf_counter()
    X = X.copy()
    y = np.asarray(y, dtype=float)
    fitted_models, results, scores = {}, {}, {}

    if len(X) >= 10:
        splits = list(TimeSeriesSplit(n_splits=2).split(X))

        for model_name, candidates in MODEL_CANDIDATES.items():
            if not candidates:
                continue

            params = candidates[0]
            fold_wape, fold_mae, fold_rmse = [], [], []

            for train_idx, val_idx in splits:
                model = build_model(model_name, params)
                model.fit(X.iloc[train_idx], y[train_idx])
                pred = model.predict(X.iloc[val_idx])

                fold_wape.append(wape(y[val_idx], pred))
                fold_mae.append(mean_absolute_error(y[val_idx], pred))
                fold_rmse.append(np.sqrt(mean_squared_error(y[val_idx], pred)))

            scores[model_name] = float(np.mean(fold_wape))
            results[model_name] = {
                "WAPE": round(float(np.mean(fold_wape)) * 100, 2),
                "MAE": round(float(np.mean(fold_mae)), 2),
                "RMSE": round(float(np.mean(fold_rmse)), 2),
                "params": params,
            }

    else:
        for model_name, candidates in MODEL_CANDIDATES.items():
            if not candidates:
                continue

            params = candidates[0]
            model = build_model(model_name, params)
            model.fit(X, y)
            pred = model.predict(X)

            metrics = evaluate(y, pred)
            scores[model_name] = metrics["WAPE"] / 100
            results[model_name] = {**metrics, "params": params}
            fitted_models[model_name] = model

    if not scores:
        raise ValueError("No forecasting models are configured.")

    best_name = min(scores, key=scores.get)
    best_params = MODEL_CANDIDATES[best_name][0]

    if best_name not in fitted_models:
        best_model = build_model(best_name, best_params)
        best_model.fit(X, y)
        fitted_models[best_name] = best_model

    elapsed = time.perf_counter() - started
    print(
        f"[TUNING] best={best_name} "
        f"WAPE={results[best_name]['WAPE']}% | "
        f"MAE={results[best_name]['MAE']} | "
        f"RMSE={results[best_name]['RMSE']} | "
        f"compared={list(scores)} | "
        f"{elapsed:.1f}s"
    )

    # --- MLflow tracking (non-blocking: failures must not break forecasting) ---
    mlflow_run_id = None
    if is_enabled():
        try:
            mlflow_run_id = log_training_run(
                dataset_key=dataset_key or "unknown",
                rows=int(len(X)),
                feature_count=int(X.shape[1]),
                results=results,
                best_name=best_name,
                best_params=best_params,
                best_model=fitted_models[best_name],
                X_train=X,
                y_train=y,
            )
        except Exception as exc:
            print(f"[MLFLOW] logging failed (non-fatal): {type(exc).__name__}: {exc}")

    # --- Prometheus metrics (non-blocking) ---
    try:
        from app.monitoring import metrics as _metrics

        _metrics.model_training_duration_seconds.labels(model=best_name).observe(elapsed)
        _metrics.model_trainings_total.labels(model=best_name, source="cache_miss").inc()
        if mlflow_run_id:
            _metrics.mlflow_runs_total.labels(
                best_model=best_name,
                registered=str(is_enabled()).lower(),
            ).inc()
    except Exception:
        pass

    return {
        "best_model": best_name,
        "best_params": best_params,
        "models": {best_name: fitted_models[best_name]},
        "metrics": results,
        "mlflow_run_id": mlflow_run_id,
    }
