import numpy as np

from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import TARGET_COL, MULTIVARIATE_COLS, TIME_STEPS


def walk_forward_statistical(
    df_train,
    df_test
):

    history_target = df_train[TARGET_COL].tolist()

    history_multi = (
        df_train[MULTIVARIATE_COLS]
        .values
        .tolist()
    )

    arima_preds = []
    sarima_preds = []
    var_preds = []

    for t in range(len(df_test)):

        if t >= TIME_STEPS:

            arima_model = ARIMA(
                history_target,
                order=(7, 1, 1)
            )

            arima_fit = arima_model.fit()

            arima_preds.append(
                arima_fit.forecast()[0]
            )

            sarima_model = SARIMAX(
                history_target,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            )

            sarima_fit = sarima_model.fit(
                disp=False
            )

            sarima_preds.append(
                sarima_fit.forecast()[0]
            )

            var_model = VAR(
                np.array(history_multi)
            )

            var_fit = var_model.fit(
                maxlags=7
            )

            forecast = var_fit.forecast(
                np.array(history_multi)[-var_fit.k_ar:],
                steps=1
            )

            var_preds.append(forecast[0][0])

        history_target.append(
            df_test.iloc[t][TARGET_COL]
        )

        history_multi.append(
            df_test.iloc[t][MULTIVARIATE_COLS]
            .values
            .tolist()
        )

    return {
        "ARIMA": np.array(arima_preds),
        "SARIMA": np.array(sarima_preds),
        "VAR": np.array(var_preds)
    }