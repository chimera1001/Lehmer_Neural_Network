import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Lehmer Transform Layer
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

# Standardize inputs for Lehmer range
def standardize_inputs_to_lehmer_range(inputs):
    min_val = tf.constant(np.exp(-1), dtype=tf.float32)
    max_val = tf.constant(np.exp(1), dtype=tf.float32)
    inputs_min = tf.reduce_min(inputs, axis=1, keepdims=True)
    inputs_max = tf.reduce_max(inputs, axis=1, keepdims=True)
    standardized = (inputs - inputs_min) / (inputs_max - inputs_min) * (max_val - min_val) + min_val
    return standardized

# Tiny ImageNet DataLoader
def load_tiny_imagenet(data_dir, batch_size=128, image_size=(64, 64)):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
    )

    return train_generator, val_generator

# Training and Evaluation
def train_and_evaluate_tiny_imagenet(data_dir, units=128, epochs=50, batch_size=128):
    print("\nTraining on Tiny ImageNet dataset...")

    # Load data
    train_gen, val_gen = load_tiny_imagenet(data_dir, batch_size)

    num_classes = 200

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        # tf.keras.layers.Dense(4096, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(4096, activation='relu'),
        # tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
        LehmerTransformLayer(units=units),
        tf.keras.layers.Lambda(standardize_inputs_to_lehmer_range),
        LehmerTransformLayer(units=units),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.5
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"Validation Accuracy: {test_accuracy:.4f}")
    print(f"Validation Loss: {test_loss:.4f}")

# Train and evaluate on Tiny ImageNet dataset
data_dir = "./Data/tiny-imagenet-200/tiny-imagenet-200"
train_and_evaluate_tiny_imagenet(data_dir, units=4, epochs=10, batch_size=128)
