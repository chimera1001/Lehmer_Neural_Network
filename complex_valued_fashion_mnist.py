import tensorflow as tf
import numpy as np


class ComplexLehmerTransformLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ComplexLehmerTransformLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.connection_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True,
            name="connection_weights"
        )
        self.s_real = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True,
            name="s_real"
        )
        self.s_imag = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            trainable=True,
            name="s_imag"
        )

    def call(self, inputs):
        s = tf.complex(self.s_real, self.s_imag)
        positive_weights = tf.math.log(1 + tf.math.exp(self.connection_weights))
        positive_weights = tf.complex(positive_weights, tf.zeros_like(positive_weights))
        positive_weights = tf.expand_dims(positive_weights, axis=0)
        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs_expanded = tf.expand_dims(inputs, axis=-1)
        inputs_power_s = tf.pow(inputs_expanded, s)
        inputs_power_s_minus_1 = tf.pow(inputs_expanded, s - 1)
        numerator = tf.reduce_sum(positive_weights * inputs_power_s, axis=1)
        denominator = tf.reduce_sum(positive_weights * inputs_power_s_minus_1, axis=1)
        lehmer_output = numerator / denominator
        return lehmer_output


class CombineComplexParts(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CombineComplexParts, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.1),
            trainable=True,
            name="alpha"
        )
        self.beta = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            trainable=True,
            name="beta"
        )
        self.gamma = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name="gamma"
        )

    def call(self, inputs):
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        combined = self.alpha * real_part + self.beta * imag_part + self.gamma
        return tf.reshape(combined, shape=(-1, self.units))


def standardize_inputs_to_lehmer_range(inputs):
    min_val = tf.constant(np.exp(-1), dtype=tf.float32)
    max_val = tf.constant(np.exp(1), dtype=tf.float32)
    inputs_min = tf.reduce_min(inputs, axis=1, keepdims=True)
    inputs_max = tf.reduce_max(inputs, axis=1, keepdims=True)
    standardized = (inputs - inputs_min) / (inputs_max - inputs_min) * (max_val - min_val) + min_val
    return standardized


def train_and_evaluate_fashion_mnist(units=128, epochs=20, batch_size=64):
    print("\nTraining on Fashion-MNIST dataset...")

    # Load Fashion-MNIST dataset
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
        
        
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.1),
        
        # tf.keras.layers1.RandomFlip(0.5),
        
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.15),
        
        # tf.keras.layers.RandomFlip(0.1),
        
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
        ComplexLehmerTransformLayer(units=units),
        CombineComplexParts(units=units),
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
    print(f"Test Accuracy on Fashion-MNIST: {test_accuracy:.2f}")
    print(f"Test Loss on Fashion-MNIST: {test_loss:.2f}")


# Train and evaluate on Fashion-MNIST
train_and_evaluate_fashion_mnist(units=4, epochs=10, batch_size=32)
