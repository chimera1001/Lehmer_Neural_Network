import tensorflow as tf
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class ComplexLehmerTransformLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ComplexLehmerTransformLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.connection_weights = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(1),
            regularizer=tf.keras.regularizers.l2(1e-4),
            trainable=True,
            name="connection_weights"
        )
        self.s_real = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=1.0),
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
        # Trainable coefficients for real and imaginary parts
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
        # Intercept term
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
        return tf.reshape(combined, shape=(-1, self.units))  # Flatten to match dense layer
    


class ModulusComplexParts(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ModulusComplexParts, self).__init__(**kwargs)

    def call(self, inputs):
        # Compute modulus of the complex input
        real_part = tf.math.real(inputs)
        imag_part = tf.math.imag(inputs)
        modulus = tf.sqrt(tf.square(real_part) + tf.square(imag_part))
        return modulus


def standardize_inputs_to_lehmer_range(inputs):
    min_val = tf.constant(np.exp(-1), dtype=tf.float32)
    max_val = tf.constant(np.exp(1), dtype=tf.float32)
    inputs_min = tf.reduce_min(inputs, axis=1, keepdims=True)
    inputs_max = tf.reduce_max(inputs, axis=1, keepdims=True)
    standardized = (inputs - inputs_min) / (inputs_max - inputs_min) * (max_val - min_val) + min_val
    return standardized


def load_dataset(name):
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X, y = data.data, data.target
    return X, y


def train_and_evaluate(dataset_name, units=4, epochs=100, batch_size=4):
    print(f"\nTraining on {dataset_name} dataset...")
    X, y = load_dataset(dataset_name)

    # scaler = MinMaxScaler((np.exp(-1),np.exp(1)))
    # X = scaler.fit_transform(X)

    # One-hot encode labels
    num_classes = len(np.unique(y))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Stratified K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, val_index in kf.split(X, y.argmax(axis=1)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Build the model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            
            tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
            ComplexLehmerTransformLayer(units=units),
            CombineComplexParts(units=units),

            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        # Compile the model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1, decay_steps=epochs, decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

        # Evaluate the model
        _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)

    # Report results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Accuracy Standard Deviation: {std_accuracy:.2f}")


# Run the model on different datasets
# datasets = ["iris", "wine", "breast_cancer", "digits"]
# for dataset in datasets:
#     train_and_evaluate(dataset_name=dataset)
train_and_evaluate(dataset_name='iris', units=3, epochs=50, batch_size=4)
