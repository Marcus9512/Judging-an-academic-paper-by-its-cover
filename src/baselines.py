import numpy as np
import logging
import pathlib
import argparse
import sys
from src.Tools.open_review_dataset import Mode
from comet_ml import Experiment
from datetime import datetime
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from enum import Enum

EXPERIMENT_LAUNCH_TIME = datetime.now()
LOGGER_NAME = "baselines"

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="rZZFwjbEXYeYOP5J0x9VTUMuf",
                        project_name="dd2430", workspace="dd2430")


class Model(Enum):
    LogisticRegression = "logistic-regression"
    RandomForest = "random-forest"

    def __str__(self):
        return self.value


model_map = {
    Model.LogisticRegression.value: LogisticRegression(solver='saga', verbose=10),
    Model.RandomForest.value: RandomForestClassifier()
}


def load_data(path_to_meta: str, mode: Mode, width: int, height: int):
    """This will load the whole dataset into memory

    This assumes that the preprocessing to generate the specified
    mode already have been executed.

    Args:
        path_to_meta: Path to the meta file of the dataset
        mode: What mode of the dataset to load
        width: The width of the preprocessed paper
        height: The height of the preprocessed paper
    """
    logger = logging.getLogger(LOGGER_NAME)

    path_to_meta = pathlib.Path(path_to_meta)
    if not path_to_meta.is_file():
        logger.fatal(f"{path_to_meta} is not a valid file")
        sys.exit(1)

    meta = pd.read_csv(path_to_meta)

    binary_blob_paths = meta.paper_path.apply(lambda x: f"{path_to_meta.parent}/{x}-{mode.value}-{width}-{height}.npy")

    binary_blobs, labels = [], []
    for path, label in zip(binary_blob_paths, meta.accepted):
        try:
            binary_blobs.append(np.load(path))
            labels.append(float(label))
        except FileNotFoundError:
            logger.warning(f"{path} does not exist, skipping..")

    return np.array(binary_blobs), np.array(labels)


def train_test_split(binary_blobs: np.ndarray, labels: np.ndarray, train_proportion: float):
    """Splits the dataset into one train split and one test split

    Args:
        binary_blobs: An array of binary blobs
        labels: An array of labels
        train_proportion: Proportion of samples to be in train split

    Returns:
        train_binary_blobs, train_labels, test_binary_blobs, test_labels
    """
    is_train = np.random.rand(labels.size) < train_proportion

    train_binary_blobs = binary_blobs[is_train]
    train_labels = labels[is_train]

    is_test = ~is_train
    test_binary_blobs = binary_blobs[is_test]
    test_labels = labels[is_test]

    return train_binary_blobs.astype(np.float16), train_labels.astype(np.float16), \
           test_binary_blobs.astype(np.float16), test_labels.astype(np.float16)


def evaluate(name: str, predictions: np.ndarray, labels: np.ndarray):
    logger = logging.getLogger(LOGGER_NAME)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)

    logger.info(f"{name} -- accuracy: {accuracy} -- recall: {recall} -- precision: {precision} ")

    # Log to comet
    experiment.log_metric(f"{name} - accuracy", accuracy)
    experiment.log_metric(f"{name} - recall", recall)
    experiment.log_metric(f"{name} - precision", precision)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(LOGGER_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_meta", type=str, help="Path to meta file", required=True)
    parser.add_argument("--mode", type=Mode, choices=list(Mode), required=True)
    parser.add_argument("--model", type=Model, choices=list(Model), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)

    args = parser.parse_args()

    # For deterministic splitting
    np.random.seed(1337)

    timestamp = time.time()

    binary_blobs, labels = load_data(args.path_to_meta, mode=args.mode, width=args.width, height=args.height)

    flat_binary_blobs = binary_blobs.reshape(len(binary_blobs), -1).astype(np.float16)

    logger.info("Splitting..")
    flat_train_binary_blobs, train_labels, flat_test_binary_blobs, test_labels = train_test_split(
        binary_blobs=flat_binary_blobs,
        labels=labels,
        train_proportion=0.8)

    logger.info(f"Fitting {args.model.value}..")

    # Sets experiment name in comet ui
    experiment.set_name(args.model.value)

    model = model_map[args.model.value]
    model.fit(X=flat_train_binary_blobs, y=train_labels)

    logger.info("Predicting..")
    train_pred = model.predict(flat_train_binary_blobs)
    test_pred = model.predict(flat_test_binary_blobs)

    print("Train_pred", train_pred)
    print("Test_pred", test_pred)
    print("Test_labels", test_labels)

    logger.info("Evaluating..")
    evaluate("train", predictions=train_pred, labels=train_labels)
    evaluate("test", predictions=test_pred, labels=test_labels)

    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
