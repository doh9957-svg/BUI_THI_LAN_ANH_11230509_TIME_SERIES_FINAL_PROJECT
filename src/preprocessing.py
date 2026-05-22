import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from config import TRAIN_PATH, TEST_PATH


def load_data(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH
):
    df_train = pd.read_csv(
        train_path,
        parse_dates=["date"],
        index_col="date"
    )

    df_test = pd.read_csv(
        test_path,
        parse_dates=["date"],
        index_col="date"
    )

    return df_train, df_test


def handle_outliers_iqr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
):
    train_clean = df_train.copy()
    test_clean = df_test.copy()

    for col in df_train.columns:

        q1 = df_train[col].quantile(0.25)
        q3 = df_train[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        train_clean[col] = np.where(
            (train_clean[col] < lower) |
            (train_clean[col] > upper),
            np.nan,
            train_clean[col]
        )

        train_clean[col] = (
            train_clean[col]
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )

        test_clean[col] = np.where(
            (test_clean[col] < lower) |
            (test_clean[col] > upper),
            np.nan,
            test_clean[col]
        )

        test_clean[col] = (
            test_clean[col]
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )

    return train_clean, test_clean


def scale_datasets(
    X_train,
    X_test,
    y_train,
    y_test
):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)
    )

    y_test_scaled = scaler_y.transform(
        y_test.values.reshape(-1, 1)
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        scaler_X,
        scaler_y
    )