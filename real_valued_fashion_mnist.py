import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class LehmerTransformLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LehmerTransformLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.connection_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(1),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True,
            name="connection_weights"
        )
        self.s = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=1.0),
            trainable=True,
            name="suddency"
        )

    def call(self, inputs):
        positive_weights = tf.math.log(1 + tf.math.exp(self.connection_weights))
        numerator = tf.reduce_sum(positive_weights * tf.pow(inputs[:, :, None], self.s), axis=1)
        denominator = tf.reduce_sum(positive_weights * tf.pow(inputs[:, :, None], self.s - 1), axis=1)
        lehmer_output = numerator / denominator
        return lehmer_output


def standardize_inputs_to_lehmer_range(inputs):
    min_val = tf.constant(np.exp(-1), dtype=tf.float32)
    max_val = tf.constant(np.exp(1), dtype=tf.float32)
    inputs_min = tf.reduce_min(inputs, axis=1, keepdims=True)
    inputs_max = tf.reduce_max(inputs, axis=1, keepdims=True)
    standardized = (inputs - inputs_min) / (inputs_max - inputs_min) * (max_val - min_val) + min_val
    return standardized


def train_and_evaluate_fashion_mnist(units=64, epochs=20, batch_size=64):
    print("\nTraining on Fashion MNIST dataset...")

    # Load Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Reshape and normalize the images
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # One-hot encode labels
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Build the model
    model = tf.keras.Sequential([
        # tf.keras.layers.RandomFlip(),
        # tf.keras.layers.RandomRotation(0.1),
        # tf.keras.layers.RandomZoom(0.1),
        
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # tf.keras.layers.RandomFlip(),
        # tf.keras.layers.RandomRotation(0.1),
        # tf.keras.layers.RandomZoom(0.1),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        
        # tf.keras.layers.Dropout(0.4),
        
        
        tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
        LehmerTransformLayer(units=units),
        
        tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
        LehmerTransformLayer(units=units),
        
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # Compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-1, decay_steps=100, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")


# Train and evaluate on Fashion MNIST dataset
train_and_evaluate_fashion_mnist(units=4, epochs=10, batch_size=64)


