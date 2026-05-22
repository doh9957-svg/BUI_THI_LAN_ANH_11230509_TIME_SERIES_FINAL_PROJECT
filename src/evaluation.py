import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import OUTPUT_DIR


def evaluate_predictions(
    predictions,
    actual
):

    rows = []

    for name, preds in predictions.items():

        if name == "Actual":
            continue

        # ==========================================
        # ALIGN LENGTHS
        # ==========================================

        min_length = min(
            len(actual),
            len(preds)
        )

        actual_aligned = actual[-min_length:]
        preds_aligned = preds[-min_length:]

        # ==========================================
        # METRICS
        # ==========================================

        mae = mean_absolute_error(
            actual_aligned,
            preds_aligned
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual_aligned,
                preds_aligned
            )
        )

        r2 = r2_score(
            actual_aligned,
            preds_aligned
        )

        rows.append({
            "Model": name,
            "Samples": min_length,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return leaderboard


def save_model_objects(objects):

    for name, obj in objects.items():

        if obj is None:
            continue

        # ==========================================
        # DEEP LEARNING MODELS
        # ==========================================

        if name in [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ]:

            obj.save(
                OUTPUT_DIR / f"{name}.h5"
            )

        # ==========================================
        # SKLEARN/XGBOOST MODELS
        # ==========================================

        else:

            joblib.dump(
                obj,
                OUTPUT_DIR / f"{name}.pkl"
            )