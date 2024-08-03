import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_evaluation_results(y_test, y_pred, output_path):
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    np.savez(output_path, report=report, confusion_matrix=cm)
