from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout


def build_cnn(input_shape, cfg):

    m = cfg["model"]["cnn"]

    model = Sequential([

        Conv1D(
            filters=m["filters"][0],
            kernel_size=m["kernels"][0],
            activation="relu",
            padding="same",
            input_shape=input_shape
        ),

        MaxPooling1D(2),

        Conv1D(
            filters=m["filters"][1],
            kernel_size=m["kernels"][1],
            activation="relu",
            padding="same"
        ),

        MaxPooling1D(2),

        Conv1D(
            filters=m["filters"][2],
            kernel_size=m["kernels"][2],
            activation="relu",
            padding="same"
        ),

        GlobalAveragePooling1D(),

        Dense(m["dense"], activation="relu"),
        Dropout(m["dropout"]),
        Dense(1)
    ])

    return model