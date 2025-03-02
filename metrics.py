from sklearn.metrics import jaccard_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import numpy as np

from models.dataset import Dataset
from models.learner.meta_learner import MetaLearner
from models.parameters import Parameters
from sklearn.metrics import recall_score

def plot_confusion_matrix(
  confusion_matrix: np.ndarray,
  labels: np.ndarray
):
  # We will assume that there are 11 labels, perhaps update in the future to be dynamic
  rows = 3
  columns = 4

  fig, axes = plt.subplots(rows, columns, figsize=(25, 11))
  axes = axes.ravel()

  print("shape of the confusion matrix: ", confusion_matrix.shape)
  print("shape of the labels: ", labels.shape)

  for index in range(11):
    print("index: ", index)
    disp = ConfusionMatrixDisplay(confusion_matrix[index], display_labels=[0, 1])
    disp.plot(ax=axes[index], values_format='.4g')
    disp.ax_.set_title(f"Class: {labels[index]}")

    # If the index is the last one, we remove the X axis label
    if index < len(axes) - columns:
      disp.ax_.set_xlabel('')

    # If the index is not divisible by the number of columns, we remove the Y axis label
    if index % columns != 0:
      disp.ax_.set_ylabel('')

    disp.im_.colorbar.remove()

  plt.subplots_adjust(wspace=0.10, hspace=0.1)
  fig.colorbar(disp.im_, ax=axes)
  plt.show()

def score_training(predictions: np.array, targets: np.array):
  """Takes the predictions per epoch stored when training the Meta-Learner and creates the curve of
  Jaccard score for the training and test data capture while training the model.
  """
  return list(map(lambda pred: jaccard_score(targets, pred, average='samples', zero_division=0), predictions))

def score_model(params: Parameters, meta_learner: MetaLearner, verbose: bool = False):
  if verbose:
    print("[INFO] Scoring Meta Learner model using the model data provided...")

  # Load the datasets from the path provided in the parameters file
  dataset_test = Dataset()
  dataset_test.load(params.dataset.test)

  dataset_training = Dataset()
  dataset_training.load(params.dataset.train)

  if verbose:
    print("[INFO] information of the dataset loaded: ")
    print("\tName: ", dataset_test.name)
    print("\tSize: ", dataset_test.dataframe.shape)
    print("\tColumns: ", dataset_test.dataframe.columns)
    print("\tHead: \n", dataset_test.dataframe.head(), "\n")

  # Pre-process the dataset for the verification phase
  dataset_test.pre_process()
  dataset_training.pre_process()

  if verbose:
    print("[INFO] information of the dataset pre-processed: ")
    print("\tColumns: ", dataset_test.dataframe_processed.columns)
    print("\tHead: \n", dataset_test.dataframe_processed.head(), "\n")

    print("[INFO] Accuracy over epochs data is generated...")
    print("[INFO] information of the current data per epoch...")
    print("\tPredictions per epoch length: ", len(meta_learner.predictions_per_epoch))
    print("\tPredictions per epoch element shape: ", meta_learner.predictions_per_epoch[0].shape)

    print("\n[INFO] Generating different metrics for the model...")

  # Generate the learning curve for the Training and Test set
  jaccard_training = score_training(meta_learner.predictions_per_epoch, dataset_training.dataframe_processed["target"].to_list())
  # TODO: add support for this jacard score accross the epochs with test data
  # jaccard_test = score_training(meta_learner.predictions_per_epoch, dataset_test.dataframe_processed["target"].to_list())

  if verbose:
    print("[INFO] Jaccard score for the training data: ", jaccard_training)
    # print("[INFO] Jaccard score for the test data: ", jaccard_test)

  # Generating several different metrics for the model
  predictions = meta_learner.full_predict(dataset_test.dataframe_processed["text"].to_list())

  # TODO: print graphics (plot_confusion_matrix function)
  # Confusion matrix
  # print(params.dataset.label_column.shape)
  # confusion_matrices = multilabel_confusion_matrix(
  #   dataset_test.dataframe_processed["target"].to_list(),
  #   predictions
  # )
  # plot_confusion_matrix(confusion_matrices, params.dataset.label_column)

  full_report = classification_report(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    target_names=params.dataset.label_column,
    zero_division=0
  )

  if verbose:
    print("[INFO] Classification report of the dataset: ")
    print(full_report)