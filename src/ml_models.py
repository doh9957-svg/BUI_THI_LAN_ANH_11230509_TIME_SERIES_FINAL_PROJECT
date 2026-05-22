import xgboost as xgb


def train_xgboost(
    X_train,
    y_train
):

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model