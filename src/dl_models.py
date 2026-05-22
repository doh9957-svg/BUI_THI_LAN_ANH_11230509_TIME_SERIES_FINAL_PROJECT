from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GRU,
    LSTM,
    SimpleRNN
)
from tensorflow.keras.models import Sequential


def build_dl_models(input_shape):

    return {

        "SimpleRNN": Sequential([
            SimpleRNN(
                32,
                activation="relu",
                input_shape=input_shape
            ),
            Dropout(0.1),
            Dense(1)
        ]),

        "LSTM": Sequential([
            LSTM(
                50,
                activation="relu",
                input_shape=input_shape
            ),
            Dropout(0.1),
            Dense(1)
        ]),

        "GRU": Sequential([
            GRU(
                50,
                activation="relu",
                input_shape=input_shape
            ),
            Dropout(0.1),
            Dense(1)
        ])
    }


def train_dl_model(
    model,
    X_train,
    y_train
):

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
        callbacks=[
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    return model