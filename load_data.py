import numpy as np
import pandas as pd
import tensorflow as tf
import logging

def load_data():
    logging.info("Loading data....")
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full),(X_test, y_test) = mnist.load_data()
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


    X_test = X_test / 255.

    return X_train, y_train, X_valid, y_valid, X_test, y_test