from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import learning_curve

import numpy as np

from models.dataset import Dataset
from models.learner.meta_learner import MetaLearner
from models.parameters import Parameters


def score_training(predictions: np.array, targets: np.array):

  # jaccard = 
  return list(map(lambda pred: jaccard_score(targets, pred, average='samples', zero_division=0), predictions))

def score_model(params_path: str, meta_learner: MetaLearner, verbose: bool = False):
  # Load the parameters from the file path passed as argument to the execution
  params = Parameters.from_json(params_path)

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
  # TODO: add support for this jaccard score accross the epochs with test data
  # jaccard_test = score_training(meta_learner.predictions_per_epoch, dataset_test.dataframe_processed["target"].to_list())

  if verbose:
    print("[INFO] Jaccard score for the training data: ", jaccard_training)
    # print("[INFO] Jaccard score for the test data: ", jaccard_test)

  # Generating several different metrics for the model
  # predictions_test = meta_learner.generate_embeddings(dataset_test.dataframe_processed["text"].to_list())
  # predictions_test_processed = meta_learner.concatenate_predictions
  predictions = meta_learner.full_predict(dataset_test.dataframe_processed["text"].to_list())

  # PRECISION 
  precision_macro = precision_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='macro'
  )
  precision_micro = precision_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='micro'
  ) 

  # RECALL
  recall_macro = recall_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='macro'
  )
  recall_micro = recall_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='micro'
  )

  # F1 SCORE
  f1_macro = f1_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='macro'
  )
  f1_micro = f1_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='micro'
  )

  # JACCARD
  jaccard_avg = jaccard_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average='samples',
    zero_division=0
  )
  jaccard = jaccard_score(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions,
    average=None,
    zero_division=0
  )

  if verbose:
    print("[INFO] Precision score for the test data: ")
    print("\tMacro: ", precision_macro)
    print("\tMicro: ", precision_micro)

    print("[INFO] Recall score for the test data: ")
    print("\tMacro: ", recall_macro)
    print("\tMicro: ", recall_micro)

    print("[INFO] F1 score for the test data: ")
    print("\tMacro: ", f1_macro)
    print("\tMicro: ", f1_micro)

    print("[INFO] Jaccard score for the test data: ")
    print("\tAverage: ", jaccard_avg)
    # print("\tIndividual: ", jaccard)

  # Confusion matrix

  confusion_matrix = multilabel_confusion_matrix(
    dataset_test.dataframe_processed["target"].to_list(),
    predictions
  )