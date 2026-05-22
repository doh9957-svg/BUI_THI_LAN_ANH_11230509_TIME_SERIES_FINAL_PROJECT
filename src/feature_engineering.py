import numpy as np
import pandas as pd

from config import TARGET_COL


def create_features(df: pd.DataFrame):

    df = df.copy()

    # ==========================================
    # DATE FEATURES
    # ==========================================

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter

    # ==========================================
    # LAG FEATURES
    # ==========================================

    df["meantemp_lag_1"] = (
        df[TARGET_COL].shift(1)
    )

    df["meantemp_lag_2"] = (
        df[TARGET_COL].shift(2)
    )

    df["meantemp_lag_7"] = (
        df[TARGET_COL].shift(7)
    )

    # ==========================================
    # ROLLING FEATURES
    # ==========================================

    df["meantemp_roll_mean_3"] = (
        df[TARGET_COL]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["meantemp_roll_mean_7"] = (
        df[TARGET_COL]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["meantemp_roll_std_7"] = (
        df[TARGET_COL]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    return df.dropna()


def create_3d_dataset(
    X,
    y,
    time_steps=7
):

    Xs = []
    ys = []

    for i in range(len(X) - time_steps):

        Xs.append(
            X[i:i + time_steps]
        )

        ys.append(
            y[i + time_steps]
        )

    return (
        np.array(Xs),
        np.array(ys)
    )