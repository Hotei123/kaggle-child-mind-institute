import tensorflow as tf
import numpy as np

shape_ts = (3, 5)

# Generator function to yield data
def data_generator():
    for _ in range(1000):  # Generate 1000 samples
        # Generate random input features and target
        inputs = np.random.rand(15).astype("float32").reshape(shape_ts)  # 10 features
        target = np.sum(inputs)  # Simple target as the sum of inputs
        yield inputs, target


def train_minimal():
    # Create the dataset using tf.data.Dataset.from_generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(shape_ts, ())  # Shapes for inputs and targets
    )

    # Define a simple model
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    #     tf.keras.layers.Dense(16, activation="relu"),
    #     tf.keras.layers.Dense(1)  # Output layer for regression
    # ])

    input = tf.keras.layers.Input(shape=shape_ts)
    x = tf.keras.layers.Dense(32, activation="relu")(input)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    # Compile the model
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Prepare the dataset for training
    train_dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    # Train the model
    model.fit(train_dataset, epochs=5)


if __name__ == '__main__':
    train_minimal()
