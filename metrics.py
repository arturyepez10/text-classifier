from sklearn.metrics import jaccard_score, multilabel_confusion_matrix, classification_report, f1_score
import numpy as np

from models.dataset import Dataset
from models.learner.meta_learner import MetaLearner
from models.parameters import Parameters
from sklearn.metrics import recall_score

def show_confusion_matrix(
  confusion_matrix: np.ndarray,
  labels: np.ndarray
):
  tn = confusion_matrix[:, 0, 0]
  tp = confusion_matrix[:, 1, 1]
  fn = confusion_matrix[:, 1, 0]
  fp = confusion_matrix[:, 0, 1]

  print()
  print(f"{'Label:':<15}TN \tTP \tFN \tFP")
  print(f"{'':<15}\t----\t----\t----\t----")
  for i, label in enumerate(labels):
    print(f"{label:<15}{tn[i]:>4} \t{tp[i]:>4} \t{fn[i]:>4} \t{fp[i]:>4}")
    print()

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

  # Confusion matrix
  confusion_matrices = multilabel_confusion_matrix(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions
  )

  # Classification report for several different metrics
  full_report = classification_report(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    target_names=params.dataset.label_column,
    zero_division=0
  )

  print("[INFO] Classification report of the dataset: ")
  print(full_report)

  print("[INFO] Confusion matrix of the dataset: ")
  show_confusion_matrix(confusion_matrices, params.dataset.label_column)

  if verbose:
    print("[INFO] We calculate some metrics for the Weak Learners...")

  # Calculate f1-score micro and macro for all weak learners
  for weak_learner in meta_learner.weak_learners:
    print(f"[INFO] Weak learner: {weak_learner.model_name}")

    weak_embeddings = weak_learner.generate_embeddings(
      dataset_test.dataframe_processed["text"].to_list()
    )
    weak_predictions = weak_learner.classifier.predict(weak_embeddings)

    # # Calculate recall
    f1micro = f1_score(
      dataset_test.dataframe_processed["target"].to_list(),
      weak_predictions,
      average='micro',
      zero_division=0
    )
    f1macro = f1_score(
      dataset_test.dataframe_processed["target"].to_list(),
      weak_predictions,
      average='macro',
      zero_division=0
    )

    print(f"\tF1 Micro: {f1micro}")
    print(f"\tF1 Macro: {f1macro}")