# src/pipeline.py

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")

from config import (
    TARGET_COL,
    TIME_STEPS,
    REPORT_DIR
)

# ==========================================
# PREPROCESSING
# ==========================================

from preprocessing import (
    load_data,
    handle_outliers_iqr,
    scale_datasets
)

# ==========================================
# EDA
# ==========================================

from eda import run_eda

# ==========================================
# FEATURE ENGINEERING
# ==========================================

from feature_engineering import (
    create_features,
    create_3d_dataset
)

# ==========================================
# VISUALIZATION
# ==========================================

from visualization import (
    plot_time_series,
    plot_correlation,
    plot_distribution,
    plot_seasonal_decompose,
    plot_forecasts
)

# ==========================================
# STATISTICAL MODELS
# ==========================================

from statistical_models import (
    walk_forward_statistical
)

# ==========================================
# MACHINE LEARNING
# ==========================================

from ml_models import (
    train_xgboost
)

# ==========================================
# DEEP LEARNING
# ==========================================

from dl_models import (
    build_dl_models,
    train_dl_model
)

# ==========================================
# EVALUATION
# ==========================================

from evaluation import (
    evaluate_predictions,
    save_model_objects
)


def align_prediction_lengths(predictions):

    lengths = []

    for name, preds in predictions.items():

        if name == "Actual":
            continue

        lengths.append(len(preds))

    min_length = min(lengths)

    aligned_predictions = {}

    for name, preds in predictions.items():

        aligned_predictions[name] = preds[-min_length:]

    return aligned_predictions, min_length


def run_pipeline():

    print("=" * 60)
    print("TIME SERIES FORECASTING PIPELINE")
    print("=" * 60)

    # ==========================================
    # LOAD DATA
    # ==========================================

    print("\nLoading data...")

    df_train, df_test = load_data()

    print(f"Train Shape: {df_train.shape}")
    print(f"Test Shape : {df_test.shape}")

    # ==========================================
    # RUN EDA
    # ==========================================

    print("\nRunning EDA...")

    run_eda(
        df_train=df_train,
        df_test=df_test
    )

    # ==========================================
    # HANDLE OUTLIERS
    # ==========================================

    print("\nHandling outliers using IQR method...")

    df_train, df_test = handle_outliers_iqr(
        df_train,
        df_test
    )

    # ==========================================
    # VISUALIZATION
    # ==========================================

    print("\nGenerating additional visualizations...")

    plot_time_series(df_train)

    plot_correlation(df_train)

    plot_distribution(df_train)

    plot_seasonal_decompose(
        df_train[TARGET_COL]
    )

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    print("\nCreating engineered features...")

    train_fe = create_features(df_train)

    test_fe = create_features(df_test)

    print(f"Train FE Shape : {train_fe.shape}")
    print(f"Test FE Shape  : {test_fe.shape}")

    feature_cols = [
        col
        for col in train_fe.columns
        if col != TARGET_COL
    ]

    X_train = train_fe[feature_cols]

    y_train = train_fe[TARGET_COL]

    X_test = test_fe[feature_cols]

    y_test = test_fe[TARGET_COL]

    print("\nFeature Columns:")

    for col in feature_cols:
        print(f"- {col}")

    # ==========================================
    # SCALING
    # ==========================================

    print("\nScaling datasets...")

    (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y
    ) = scale_datasets(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # ==========================================
    # CREATE 3D DATASETS
    # ==========================================

    print("\nCreating 3D datasets for RNN/LSTM/GRU...")

    X_train_3d, y_train_3d = create_3d_dataset(
        X_train_scaled,
        y_train_scaled,
        TIME_STEPS
    )

    X_test_3d, y_test_3d = create_3d_dataset(
        X_test_scaled,
        y_test_scaled,
        TIME_STEPS
    )

    print(f"X_train_3d Shape : {X_train_3d.shape}")
    print(f"X_test_3d Shape  : {X_test_3d.shape}")

    # ==========================================
    # TARGET VALUES
    # ==========================================

    actual_values = y_test.iloc[TIME_STEPS:].values

    eval_index = y_test.iloc[TIME_STEPS:].index

    predictions = {
        "Actual": actual_values
    }

    # ==========================================
    # STATISTICAL MODELS
    # ==========================================

    print("\nRunning Walk-Forward Statistical Models...")

    classical_preds = walk_forward_statistical(
        df_train,
        df_test
    )

    for name, preds in classical_preds.items():

        predictions[name] = np.array(preds).flatten()

    print("Statistical models completed.")

    # ==========================================
    # XGBOOST
    # ==========================================

    print("\nTraining XGBoost...")

    xgb_model = train_xgboost(
        X_train.iloc[TIME_STEPS:],
        y_train.iloc[TIME_STEPS:]
    )

    xgb_preds = xgb_model.predict(
        X_test.iloc[TIME_STEPS:]
    )

    predictions["XGBoost"] = (
        np.array(xgb_preds)
        .flatten()
    )

    print("XGBoost completed.")

    # ==========================================
    # DEEP LEARNING
    # ==========================================

    print("\nTraining Deep Learning models...")

    input_shape = (
        X_train_3d.shape[1],
        X_train_3d.shape[2]
    )

    dl_models = build_dl_models(input_shape)

    for name, model in dl_models.items():

        print(f"\nTraining {name}...")

        model = train_dl_model(
            model,
            X_train_3d,
            y_train_3d
        )

        preds_scaled = model.predict(
            X_test_3d,
            verbose=0
        )

        preds = scaler_y.inverse_transform(
            preds_scaled
        ).flatten()

        predictions[name] = preds

        print(f"{name} completed.")

    # ==========================================
    # ALIGN PREDICTION LENGTHS
    # ==========================================

    print("\nAligning prediction lengths...")

    predictions, aligned_length = align_prediction_lengths(
        predictions
    )

    actual_values = predictions["Actual"]

    eval_index = eval_index[-aligned_length:]

    print(f"Aligned Length: {aligned_length}")

    # ==========================================
    # EVALUATION
    # ==========================================

    print("\nEvaluating models...")

    leaderboard = evaluate_predictions(
        predictions,
        actual_values
    )

    print("\n")
    print("=" * 60)
    print("MODEL LEADERBOARD")
    print("=" * 60)

    print(leaderboard)

    leaderboard.to_csv(
        REPORT_DIR / "leaderboard.csv",
        index=False
    )

    # ==========================================
    # FORECAST VISUALIZATION
    # ==========================================

    print("\nGenerating forecast plots...")

    plot_forecasts(
        eval_index,
        predictions
    )

    # ==========================================
    # SAVE PREDICTIONS
    # ==========================================

    print("\nSaving forecast predictions...")

    forecast_df = pd.DataFrame({
        "date": eval_index,
        "Actual": actual_values
    })

    for name, preds in predictions.items():

        if name == "Actual":
            continue

        forecast_df[name] = preds

    forecast_df.to_csv(
        REPORT_DIR / "forecast_predictions.csv",
        index=False
    )

    # ==========================================
    # SAVE MODELS
    # ==========================================

    print("\nSaving trained models...")

    save_model_objects({
        "XGBoost": xgb_model,
        **dl_models
    })

    # ==========================================
    # FINAL SUMMARY
    # ==========================================

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated Reports:")
    print("- leaderboard.csv")
    print("- forecast_predictions.csv")
    print("- train_summary.csv")
    print("- test_summary.csv")
    print("- missing_values.csv")
    print("- adf_test.csv")

    print("\nSaved Models:")
    print("- XGBoost.pkl")
    print("- SimpleRNN.h5")
    print("- LSTM.h5")
    print("- GRU.h5")

    print("\nGenerated Visualizations:")
    print("- EDA plots")
    print("- ACF/PACF")
    print("- Correlation heatmaps")
    print("- Forecast comparison plots")
    print("- Seasonal decomposition")

    print("\nOutput Directory:")
    print(REPORT_DIR)

    print("\nPipeline finished successfully.")