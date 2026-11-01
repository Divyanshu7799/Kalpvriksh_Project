"""Train classifiers on some loads and score them on a load they have never seen."""

from collections import namedtuple

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machineguard.config import CLASS_ORDER

DUMMY = "dummy (most frequent class)"
LOGISTIC_REGRESSION = "logistic regression"

ModelResult = namedtuple("ModelResult", ["macro_f1", "confusion"])


def make_models():
    return {
        DUMMY: DummyClassifier(strategy="most_frequent"),
        LOGISTIC_REGRESSION: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
    }


def evaluate_held_out_load(window_table, feature_names, held_out_load):
    """Return {model name: ModelResult(macro_f1, confusion)} for one held-out load."""
    train = window_table[window_table["load_hp"] != held_out_load]
    test = window_table[window_table["load_hp"] == held_out_load]

    results = {}
    for name, model in make_models().items():
        model.fit(train[feature_names], train["fault_type"])
        predicted = model.predict(test[feature_names])
        macro_f1 = f1_score(test["fault_type"], predicted, labels=CLASS_ORDER, average="macro", zero_division=0)
        matrix = confusion_matrix(test["fault_type"], predicted, labels=CLASS_ORDER)
        results[name] = ModelResult(macro_f1, matrix)
    return results
