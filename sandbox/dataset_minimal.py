import tensorflow as tf
import numpy as np
import math

shape_ts = (5, 7)
shape_tab = (4,)

# Generator function to yield data
def data_generator():
    for _ in range(1000):  # Generate 1000 samples
        # Generate random input features and target
        inputs_ts = np.random.rand(math.prod(shape_ts)).astype("float32").reshape(shape_ts)  # 10 features
        inputs_tab = np.random.rand(math.prod(shape_tab)).astype("float32").reshape(shape_tab)  # 10 features
        target = np.sum(inputs_ts)  # Simple target as the sum of inputs
        yield (inputs_ts, inputs_tab), target


def train_minimal():
    # Create the dataset using tf.data.Dataset.from_generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=((shape_ts, shape_tab), ())  # Shapes for inputs and targets
    )

    input_ts = tf.keras.layers.Input(shape=shape_ts)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(input_ts)
    x = tf.keras.layers.LSTM(16)(x)
    x = tf.keras.layers.Flatten()(x)

    input_tab = tf.keras.layers.Input(shape=shape_tab)
    y = tf.keras.layers.Dense(4, activation="relu")(input_tab)

    x = tf.keras.layers.concatenate([x, y])
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[input_ts, input_tab], outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Prepare the dataset for training
    train_dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Train the model
    model.fit(train_dataset, epochs=5)


if __name__ == '__main__':
    train_minimal()
