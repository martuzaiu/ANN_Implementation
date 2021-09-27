import tensorflow as tf
import logging

def create_model(hidden1, hidden2, hidden3):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputLayer"),
          tf.keras.layers.Dense(hidden1, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(hidden2, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(hidden3, activation="softmax", name="outputLayer")
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    return model_clf