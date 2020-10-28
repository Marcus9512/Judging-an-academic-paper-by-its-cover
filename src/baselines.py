import numpy as np
import logging
import pathlib
import argparse
import sys
import os

# Fixes python path for some 
sys.path.append(os.getcwd())

from src.Tools.open_review_dataset import Mode
from comet_ml import Experiment
from datetime import datetime
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from enum import Enum
import matplotlib.pyplot as plt

EXPERIMENT_LAUNCH_TIME = datetime.now()
LOGGER_NAME = "baselines"

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="rZZFwjbEXYeYOP5J0x9VTUMuf", project_name="dd2430", workspace="dd2430")


class DimReduction(Enum):
    Nothing = "nothing"
    PCA = "pca"

    def __str__(self):
        return self.value


class Model(Enum):
    LogisticRegression = "logistic-regression"
    RandomForest = "random-forest"

    def __str__(self):
        return self.value


model_map = {
    Model.LogisticRegression.value: LogisticRegression(solver='saga', verbose=10, max_iter=3, n_jobs=3),
    Model.RandomForest.value: RandomForestClassifier()
}

dim_reduction_map = {
    DimReduction.PCA.value: IncrementalPCA(n_components=100, batch_size=200)
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

    return np.array(binary_blobs).astype(np.float16), np.array(labels)


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
    parser.add_argument("--dim_reduction", type=DimReduction, choices=list(DimReduction), default=DimReduction.Nothing)
    parser.add_argument("--comet_tag", type=str, required=True)

    args = parser.parse_args()

    # For deterministic splitting
    np.random.seed(1337)

    timestamp = time.time()

    logger.info("Loading binary blobs..")
    binary_blobs, labels = load_data(args.path_to_meta, mode=args.mode, width=args.width, height=args.height)

    logger.info("Reshaping binary blobs..")
    flat_binary_blobs = binary_blobs.reshape(len(binary_blobs), -1).astype(np.float16)

    logger.info("Normalizing binary blobs..")
    # Normalize image
    flat_binary_blobs = flat_binary_blobs / flat_binary_blobs.max()

    # To save memory we can now del old data
    del binary_blobs

    logger.info("Splitting..")
    flat_train_binary_blobs, train_labels, flat_test_binary_blobs, test_labels = train_test_split(
        binary_blobs=flat_binary_blobs,
        labels=labels,
        train_proportion=0.8)

    # To save memory we can now del old data
    del flat_binary_blobs
    del labels

    # Sets experiment name in comet ui
    experiment.set_name(args.model.value)
    experiment.add_tag(args.comet_tag)
    experiment.add_tag(args.mode.value)
    experiment.add_tag(args.dim_reduction.value)

    logger.info(f"Train size: {flat_train_binary_blobs.shape}")

    dim_reduction = None
    if args.dim_reduction != DimReduction.Nothing:
        dim_reduction = dim_reduction_map[args.dim_reduction.value]

        logger.info(f"Fitting {args.dim_reduction.value}..")
        dim_reduction.fit(flat_train_binary_blobs)

        flat_train_binary_blobs = dim_reduction.transform(flat_train_binary_blobs)
        flat_test_binary_blobs = dim_reduction.transform(flat_test_binary_blobs)

    logger.info(f"Fitting {args.model.value}..")
    model = model_map[args.model.value]
    model.fit(X=flat_train_binary_blobs, y=train_labels)

    logger.info("Predicting..")
    train_pred = model.predict(flat_train_binary_blobs)
    test_pred = model.predict(flat_test_binary_blobs)

    logger.info("Evaluating..")
    evaluate("train", predictions=train_pred, labels=train_labels)
    evaluate("test", predictions=test_pred, labels=test_labels)

    if args.model == Model.LogisticRegression and args.dim_reduction == DimReduction.PCA:
        n = 4
        s = model.coef_[0].argsort()

        top_n_smallest = s[:n]
        top_n_largest = s[-n:]

        top_n_best_eigen_papers_for_acceptance = dim_reduction.components_[top_n_largest]
        top_n_best_eigen_papers_for_rejection = dim_reduction.components_[top_n_smallest]

        shape = (args.width, args.height)
        if args.mode == Mode.RGBBigImage:
            shape = (args.width * 2, args.height * 4, 3)

        acceptance_eigen = [paper.reshape(shape) for paper in top_n_best_eigen_papers_for_acceptance]
        rejection_eigen = [paper.reshape(shape) for paper in top_n_best_eigen_papers_for_rejection]

        # Make them into images
        acceptance_eigen = [(paper - paper.min()) / paper.max() for paper in acceptance_eigen]
        rejection_eigen = [(paper - paper.min()) / paper.max() for paper in rejection_eigen]

        for i, acceptance in enumerate(reversed(acceptance_eigen), 1):
            experiment.log_image(acceptance, name=f"{i} - acceptance")

        for i, rejection in enumerate(rejection_eigen, 1):
            experiment.log_image(rejection, name=f"{i} - rejection")

    logger.info(
        f"Execution time was {time.time() - timestamp} seconds")
