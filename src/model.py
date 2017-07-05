from .lessons import *

class PersistentModel:
    clf = None
    X_test = None
    y_test = None
    X_scaler = None

    def __init__(self, directory, templates_path, hyperparams):
        if os.path.isfile(filename):
            clf, X_test, y
        clf, X_test, y_test, X_scaler = generate_classifier(templates_path, hyperparams)
        self.svc = clf
        self.

    def load_from_files(self, directory):
