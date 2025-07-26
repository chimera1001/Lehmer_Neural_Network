import tensorflow as tf
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


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


def load_dataset(name):
    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == "adult":
        data = fetch_openml("adult", version=2, as_frame=True)
        df = data.frame
        X = df.drop(columns="class")
        y = df["class"]
        
        # Preprocess categorical and numerical features
        categorical_features = X.select_dtypes(include=["object"]).columns
        numerical_features = X.select_dtypes(include=["number"]).columns
        
        preprocess = ColumnTransformer([
            ("num", MinMaxScaler(feature_range=(np.exp(-1), np.exp(1))), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
        
        pipeline = Pipeline(steps=[("preprocess", preprocess)])
        X = pipeline.fit_transform(X)
        y = (y == ">50K").astype(int)  # Binary classification
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y


def train_and_evaluate(dataset_name, units=4, epochs=50, batch_size=4):
    print(f"\nTraining on {dataset_name} dataset...")
    X, y = load_dataset(dataset_name)
    
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # if dataset_name != "adult":  # Adult dataset is already scaled in preprocessing
    #     X = scaler.fit_transform(X)

    scaler = MinMaxScaler(feature_range=(np.exp(-1), np.exp(1)))
    if dataset_name != "adult":  # Adult dataset is already scaled in preprocessing
        X = scaler.fit_transform(X)

    num_classes = len(np.unique(y))
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, val_index in kf.split(X, y.argmax(axis=1)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            
            # tf.keras.layers.Dense(units, activation='relu'),
            
            tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
            LehmerTransformLayer(units=units),
            
            # tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
            # LehmerTransformLayer(units=units),
            
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-1,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

        _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_accuracies.append(val_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Dataset: {dataset_name}")
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Accuracy Standard Deviation: {std_accuracy:.2f}")


# Train and evaluate on Iris, Wine, Breast Cancer, and Adult datasets
datasets = ["iris", "wine", "breast_cancer", "adult"]
# for dataset in datasets:
train_and_evaluate(dataset_name="breast_cancer", units=3, epochs=50, batch_size=4)














