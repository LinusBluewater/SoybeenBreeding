from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate


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


def build_cnn_with_env(geno_shape, env_dim, cfg):
    """
    双输入模型：
    - 基因型分支：Conv1D 提取 SNP 序列特征
    - 环境分支：Dense 处理 one-hot 环境编码
    - 合并后预测表型值
    """
    m = cfg["model"]["cnn"]

    # === 基因型分支 (CNN) ===
    geno_input = Input(shape=geno_shape, name="genotype")
    x = Conv1D(
        filters=m["filters"][0],
        kernel_size=m["kernels"][0],
        activation="relu",
        padding="same"
    )(geno_input)
    x = MaxPooling1D(2)(x)

    x = Conv1D(
        filters=m["filters"][1],
        kernel_size=m["kernels"][1],
        activation="relu",
        padding="same"
    )(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(
        filters=m["filters"][2],
        kernel_size=m["kernels"][2],
        activation="relu",
        padding="same"
    )(x)
    x = GlobalAveragePooling1D()(x)
    geno_features = Dense(m["dense"], activation="relu", name="geno_dense")(x)

    # === 环境分支 ===
    env_input = Input(shape=(env_dim,), name="environment")
    env_features = Dense(16, activation="relu", name="env_dense")(env_input)

    # === 合并 ===
    merged = Concatenate(name="merge")([geno_features, env_features])
    merged = Dropout(m["dropout"])(merged)
    output = Dense(1, name="phenotype")(merged)

    model = Model(
        inputs=[geno_input, env_input],
        outputs=output
    )

    return model