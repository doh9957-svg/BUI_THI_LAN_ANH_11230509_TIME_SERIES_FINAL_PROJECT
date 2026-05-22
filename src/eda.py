import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf
)

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from config import (
    TARGET_COL,
    MULTIVARIATE_COLS,
    REPORT_DIR
)

from utils import save_plot


def run_eda(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
):

    print("=" * 60)
    print("TRAIN SET SUMMARY")
    print("=" * 60)

    print(df_train.describe().T)

    print("\n")

    print("=" * 60)
    print("TEST SET SUMMARY")
    print("=" * 60)

    print(df_test.describe().T)

    summary_train_path = REPORT_DIR / "train_summary.csv"
    summary_test_path = REPORT_DIR / "test_summary.csv"

    df_train.describe().T.to_csv(summary_train_path)
    df_test.describe().T.to_csv(summary_test_path)

    # ==========================================
    # MISSING VALUES CHECK
    # ==========================================

    print("=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing_train = df_train.isnull().sum()
    missing_test = df_test.isnull().sum()

    missing_df = pd.DataFrame({
        "Train Missing": missing_train,
        "Test Missing": missing_test
    })

    print(missing_df)

    missing_df.to_csv(
        REPORT_DIR / "missing_values.csv"
    )

    # ==========================================
    # DATA TYPES
    # ==========================================

    print("=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    dtype_df = df_train.dtypes.to_frame("dtype")

    print(dtype_df)

    dtype_df.to_csv(
        REPORT_DIR / "data_types.csv"
    )

    # ==========================================
    # TARGET SERIES VISUALIZATION
    # ==========================================

    plt.figure(figsize=(16, 5))

    plt.plot(
        df_train.index,
        df_train[TARGET_COL],
        label="Train",
        alpha=0.8,
        linewidth=1.5
    )

    plt.plot(
        df_test.index,
        df_test[TARGET_COL],
        label="Test",
        alpha=0.8,
        linewidth=1.5
    )

    plt.title(
        "Mean Temperature: Train vs Test Series",
        fontsize=16
    )

    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")

    plt.legend()

    save_plot(
        "01_train_test_series.png"
    )

    # ==========================================
    # MULTIVARIATE TIME SERIES
    # ==========================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        sharex=True
    )

    axes = axes.flatten()

    for i, col in enumerate(MULTIVARIATE_COLS):

        axes[i].plot(
            df_train.index,
            df_train[col],
            linewidth=1.2
        )

        axes[i].set_title(col)

    plt.suptitle(
        "Climate Variables Over Time",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_plot(
        "02_multivariate_series.png"
    )

    # ==========================================
    # DISTRIBUTION PLOTS
    # ==========================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )

    axes = axes.flatten()

    for i, col in enumerate(MULTIVARIATE_COLS):

        sns.histplot(
            df_train[col],
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Distribution of {col}"
        )

    save_plot(
        "03_distribution_plots.png"
    )

    # ==========================================
    # BOXPLOTS
    # ==========================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )

    axes = axes.flatten()

    for i, col in enumerate(MULTIVARIATE_COLS):

        sns.boxplot(
            y=df_train[col],
            ax=axes[i]
        )

        axes[i].set_title(
            f"Boxplot of {col}"
        )

    save_plot(
        "04_boxplots.png"
    )

    # ==========================================
    # CORRELATION MATRIX
    # ==========================================

    corr_matrix = (
        df_train[MULTIVARIATE_COLS]
        .corr()
    )

    corr_matrix.to_csv(
        REPORT_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )

    plt.title(
        "Correlation Matrix for Climate Variables",
        fontsize=15
    )

    save_plot(
        "05_correlation_matrix.png"
    )

    # ==========================================
    # SEASONAL DECOMPOSITION
    # ==========================================

    decomposition = seasonal_decompose(
        df_train[TARGET_COL],
        period=365,
        model="additive",
        extrapolate_trend="freq"
    )

    fig = decomposition.plot()

    fig.set_size_inches(14, 10)

    plt.suptitle(
        "Seasonal Decomposition of Mean Temperature",
        fontsize=16,
        y=1.02
    )

    save_plot(
        "06_seasonal_decomposition.png"
    )

    # ==========================================
    # AUTOCORRELATION PLOT
    # ==========================================

    plt.figure(figsize=(14, 4))

    pd.plotting.autocorrelation_plot(
        df_train[TARGET_COL]
    )

    plt.title(
        "Autocorrelation Plot for Mean Temperature",
        fontsize=14
    )

    save_plot(
        "07_autocorrelation_plot.png"
    )

    # ==========================================
    # ACF / PACF
    # ==========================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 4)
    )

    plot_acf(
        df_train[TARGET_COL],
        lags=40,
        ax=axes[0]
    )

    plot_pacf(
        df_train[TARGET_COL],
        lags=40,
        ax=axes[1]
    )

    axes[0].set_title(
        "Autocorrelation Function (ACF)"
    )

    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)"
    )

    save_plot(
        "08_acf_pacf.png"
    )

    # ==========================================
    # LAG PLOT
    # ==========================================

    plt.figure(figsize=(8, 6))

    pd.plotting.lag_plot(
        df_train[TARGET_COL],
        lag=7
    )

    plt.title(
        "Lag Plot for Mean Temperature (lag=7)",
        fontsize=14
    )

    save_plot(
        "09_lag_plot.png"
    )

    # ==========================================
    # ROLLING STATISTICS
    # ==========================================

    rolling_mean = (
        df_train[TARGET_COL]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    rolling_std = (
        df_train[TARGET_COL]
        .rolling(window=30, min_periods=1)
        .std()
    )

    plt.figure(figsize=(16, 5))

    plt.plot(
        df_train[TARGET_COL],
        color="lightgray",
        alpha=0.7,
        label="Original Series"
    )

    plt.plot(
        rolling_mean,
        color="blue",
        linewidth=2,
        label="30-day Rolling Mean"
    )

    plt.plot(
        rolling_std,
        color="red",
        linewidth=2,
        label="30-day Rolling Std"
    )

    plt.title(
        "Rolling Mean and Standard Deviation",
        fontsize=15
    )

    plt.legend()

    save_plot(
        "10_rolling_statistics.png"
    )

    # ==========================================
    # MONTHLY SEASONALITY
    # ==========================================

    monthly_df = df_train.copy()

    monthly_df["month"] = (
        monthly_df.index.month
    )

    plt.figure(figsize=(14, 5))

    sns.boxplot(
        x="month",
        y=TARGET_COL,
        data=monthly_df
    )

    plt.title(
        "Seasonal Distribution of Mean Temperature by Month",
        fontsize=15
    )

    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")

    save_plot(
        "11_monthly_seasonality.png"
    )

    # ==========================================
    # TRAIN VS TEST DISTRIBUTION
    # ==========================================

    plt.figure(figsize=(12, 5))

    sns.kdeplot(
        df_train[TARGET_COL],
        label="Train",
        linewidth=2
    )

    sns.kdeplot(
        df_test[TARGET_COL],
        label="Test",
        linewidth=2
    )

    plt.title(
        "Train vs Test Distribution",
        fontsize=15
    )

    plt.xlabel("Temperature (°C)")

    plt.legend()

    save_plot(
        "12_train_test_distribution.png"
    )

    # ==========================================
    # STATIONARITY TEST (ADF)
    # ==========================================

    print("=" * 60)
    print("ADF STATIONARITY TEST")
    print("=" * 60)

    adf_result = adfuller(
        df_train[TARGET_COL]
    )

    adf_output = pd.DataFrame({
        "Metric": [
            "ADF Statistic",
            "p-value",
            "Used Lags",
            "Number of Observations"
        ],
        "Value": [
            adf_result[0],
            adf_result[1],
            adf_result[2],
            adf_result[3]
        ]
    })

    print(adf_output)

    adf_output.to_csv(
        REPORT_DIR / "adf_test.csv",
        index=False
    )

    print("\nCritical Values:")

    for key, value in adf_result[4].items():

        print(f"{key}: {value:.4f}")

    if adf_result[1] < 0.05:

        print("\nResult: Series is STATIONARY")

    else:

        print("\nResult: Series is NON-STATIONARY")

    # ==========================================
    # FEATURE CORRELATION WITH TARGET
    # ==========================================

    feature_corr = (
        df_train.corr()[TARGET_COL]
        .sort_values(ascending=False)
        .to_frame("Correlation")
    )

    feature_corr.to_csv(
        REPORT_DIR / "feature_target_correlation.csv"
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        feature_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Feature Correlation with Target",
        fontsize=14
    )

    save_plot(
        "13_feature_target_correlation.png"
    )

    # ==========================================
    # SUMMARY
    # ==========================================

    print("=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Train Shape : {df_train.shape}")
    print(f"Test Shape  : {df_test.shape}")

    print("\nVariables:")

    for col in MULTIVARIATE_COLS:
        print(f"- {col}")

    print("\nEDA Outputs Generated:")
    print("- Summary statistics")
    print("- Missing value analysis")
    print("- Distribution analysis")
    print("- Boxplots")
    print("- Time series visualization")
    print("- Correlation heatmap")
    print("- Seasonal decomposition")
    print("- ACF/PACF plots")
    print("- Lag analysis")
    print("- Rolling statistics")
    print("- Monthly seasonality")
    print("- Train/Test distribution comparison")
    print("- ADF stationarity test")