from load_data import load_data
from model import create_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


def train_model(model_clf, epochs, X_train, y_train, X_valid, y_valid):
    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "ADAM"
    METRICS = ["accuracy"]
    VALIDATION = (X_valid, y_valid)
    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    logging.info("Model training")
    logging.info("-----"*10)

    history = model_clf.fit(X_train, y_train, epochs=epochs, validation_data=VALIDATION)
    model_clf.save("models/model.h5")

    return model_clf, history

def save_plot(history, file_name):
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)  
    plotPath = os.path.join(plot_dir, file_name)
    plt.savefig(plotPath)

def test_model(model_clf, X_test, y_test):
    logging.info("Model testing")
    logging.info("-----"*10)
    model_clf.evaluate(X_test, y_test)

def main(epochs):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    model = create_model(100, 35, 10)

    model, history = train_model(model, epochs, X_train, y_train, X_valid, y_valid)

    save_plot(history, "plot1")

    test_model(model, X_test, y_test)

if __name__ == '__main__':
    epochs = 5
    main(epochs)