import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

from utils import save_plot


def plot_time_series(df):

    fig, axes = plt.subplots(
        len(df.columns),
        1,
        figsize=(16, 10),
        sharex=True
    )

    palette = sns.color_palette("tab10")

    for idx, col in enumerate(df.columns):

        axes[idx].plot(
            df.index,
            df[col],
            color=palette[idx],
            linewidth=1.8
        )

        axes[idx].set_title(f"{col} over time")

    save_plot("01_time_series.png")


def plot_correlation(df):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")

    save_plot("02_correlation_matrix.png")


def plot_distribution(df):

    plt.figure(figsize=(14, 5))

    for idx, col in enumerate(df.columns):

        plt.subplot(1, len(df.columns), idx + 1)

        sns.histplot(
            df[col],
            kde=True
        )

        plt.title(col)

    save_plot("03_distributions.png")


def plot_seasonal_decompose(series):

    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=7
    )

    fig = decomposition.plot()
    fig.set_size_inches(16, 10)

    save_plot("04_seasonal_decompose.png")


def plot_forecasts(eval_index, predictions):

    plt.figure(figsize=(16, 8))

    plt.plot(
        eval_index,
        predictions["Actual"],
        label="Actual",
        linewidth=3
    )

    for name, preds in predictions.items():

        if name == "Actual":
            continue

        plt.plot(
            eval_index,
            preds,
            label=name
        )

    plt.legend()

    plt.title("Forecast Comparison")

    save_plot("05_forecast_comparison.png")