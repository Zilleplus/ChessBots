from random import shuffle
from chess import Board
import tensorflow as tf  # type: ignore
from mmEngine.value_funtions.value_function import ValueFunction  # type: ignore
import tensorflow.keras as keras  # type: ignore
import tensorflow.keras.layers as layers  # type: ignore
import tensorflow.keras.backend as K  # type: ignore
import numpy as np
from typing import Optional
from pathlib import Path
from mmEngine.database import convert
import wandb
from wandb.keras import WandbCallback


def LoadModel(file_location: Path):
    assert file_location.exists()
    return keras.models.load_model(file_location)

def CreateModel() -> keras.Model:
    # 8*8 board -> one hot encoding of 12 states
    model = keras.Sequential([
        layers.InputLayer(input_shape=(8, 8, 12)),

        # 8*8 boards
        layers.Conv2D(filters=12, kernel_size=1,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=16, kernel_size=3,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=32, kernel_size=3,
                      padding="same", activation="relu"),

        layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

        # 4*4 boards
        layers.Conv2D(filters=64, kernel_size=2,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=64, kernel_size=2,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=128, kernel_size=2,
                      padding="same", activation="relu"),

        layers.MaxPooling2D(pool_size=(2, 2), padding="same"),

        # 2*2 boards
        layers.Conv2D(filters=128, kernel_size=1,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=128, kernel_size=1,
                      padding="same", activation="relu"),
        layers.Dropout(0.5),
        layers.Conv2D(filters=128, kernel_size=1,
                      padding="same", activation="relu"),

        layers.Flatten(),
        layers.Dense(units=1, activation="tanh")
    ])

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e2,
    #     decay_steps=5,
    #     decay_rate=0.9)

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mse',
                  metrics=["MeanSquaredError"])

    return model


def TrainModel(model: keras.Model,
               dataset: list[np.ndarray],
               save_path: Optional[Path] = None,
               log_dir_board: Optional[Path] = None,
               print_gradients: bool = False):
    assert len(dataset) > 0
    batch_size: int = 1024

    X = tf.data.Dataset.from_tensors(dataset[0]["X"]).unbatch()
    for i in range(len(dataset)-1):
        X = X.concatenate(tf.data.Dataset.from_tensors(
            dataset[i + 1]["X"]).unbatch())

    X_reshaped = X \
        .map(lambda x: tf.reshape(x, (8, 8)))

    X_one_hot = X_reshaped \
        .map(lambda x: tf.one_hot(x, 12))

    Y = tf.data.Dataset.from_tensors(dataset[0]["Y"]).unbatch()
    for i in range(len(dataset)-1):
        Y = Y.concatenate(tf.data.Dataset.from_tensors(
            dataset[i+1]["Y"]).unbatch())

    Y = Y.map(lambda x: tf.cast(x, tf.float32)) \

    data = tf.data.Dataset.zip((X_one_hot, Y))\
        .shuffle(10000) \
        .batch(batch_size=batch_size)

    # Take 1000 batches of data as validation set.
    # The shuffle made sure there spread out from
    # all kings of games.
    validation_size = 1000
    validation_data = data.take(validation_size)
    train_data = data.skip(validation_size)  # .take(validation_size*10)

    callbacks: list = []

    if save_path is not None:
        print(f"Adding save callback to location: {save_path}")
        callback_save = keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True)
        callbacks.append(callback_save)

    if log_dir_board is not None:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir_board))

    wandb.init(config={"hyper": "parameter"})
    callbacks.append(WandbCallback())

    class GradCallback(keras.callbacks.Callback):
        model: keras.Model

        def __init__(self, model: keras.Model):
            self.model = model

        def on_epoch_end(self, epoch, logs):
            with tf.GradientTape() as tape:
                loss = model(model.trainable_weights)

            grad = tape.gradient(loss, model.trainable_weights)
            print(grad)

    if print_gradients:
        callbacks.append(GradCallback(model))

    model.fit(x=train_data, validation_data=validation_data,
              epochs=100, callbacks=callbacks)

    return


class NNKerasValueFunction(ValueFunction):
    def __init__(self, keras_model_location: Path):
        self.model = LoadModel(keras_model_location)

    def __call__(self, board: Board) -> float:
        tf_board = tf.data.Dataset.from_tensors(convert(board))
        tf_board = tf_board \
            .map(lambda x: tf.reshape(x, (8, 8))) \
            .map(lambda x: tf.one_hot(x, 12)) \
            .map(lambda x: tf.cast(x, tf.float32)) \
            .batch(1)

        tensor = next(iter(tf_board))
        pred = self.model(tensor, training=False)

        if board.turn:
            return float(pred)
        else:
            return -float(pred)
