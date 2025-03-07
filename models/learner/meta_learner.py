import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

from os import path, mkdir
import pickle

from models.learner import Learner
from models.parameters.learner import Classifier
from models.learner.weak_learner import WeakLearner

# TODO: Add better support for intelligent data catching / logs in training process to create the graphics
class MetaLearner(Learner):
  epochs = 2
  batch_size = 128

  def __init__(self, classifier_params: Classifier):
    super().__init__(classifier_params)

    self.weak_learners: list[WeakLearner] = []

    self.predictions_processed = np.array([])
    self.predictions_per_epoch = []

  def add_weak_learner(self, weak_learner: WeakLearner):
    self.weak_learners.append(weak_learner)

  def generate_embeddings(self, data: list | None = None):
    return np.array([weak_learner.predict(data) for weak_learner in self.weak_learners])
  
  def concatenate_predictions(self, predictions: list):
    """ Concatenate the predictions of each one of the weak learners associated to the meta learner

    What it does, is that for each prediction generated by each weak learner, it concatenates them
    into a single element of a list

    For example, given that we obtain the predictions of size 3 of N weak learners, the output will be:
    .. code-block:: python

    [
     [prediction_1_1, prediction_2_1, prediction_3_1],
     [prediction_1_2, prediction_2_2, prediction_3_2],
     ...
     [prediction_1_N, prediction_2_N, prediction_3_N]
    ]
    """
    # TODO: this commented code is a literal way to do it extracted from original thesis
    # return [
    #   np.concatenate(
    #     [
    #     embeddings[j][i] for j in range(len(embeddings))
    #     ], 
    #     axis=0
    #   ) for i in range(len(embeddings[0]))
    # ]
    return np.array([
      np.concatenate(
        [
        predictions[j][i] for j in range(len(predictions))
        ], 
        axis=0
      ) for i in range(len(predictions[0]))
    ])
  
  # TODO: this whole funciton should be optimize to obtain it in a better way
  def train_model(self, dataframe: pd.DataFrame):
    epoch = 0

    classes_qty = list(range(len(dataframe["target"][0]))) # TODO: this has to be a list of number from 0 to the max number of classes (-1). OPTIMIZE

    predictions = [weak_learner.predict() for weak_learner in self.weak_learners]
    self.predictions_processed = self.concatenate_predictions(predictions)

    train_samples = len(self.predictions_processed)
    while epoch < self.epochs:
      mini_batch_index = 0
      while True:
        self.classifier.partial_fit(self.predictions_processed, dataframe["target"].to_list(), classes=classes_qty)
        mini_batch_index += self.batch_size

        if mini_batch_index >= train_samples:
          break

      self.predictions_per_epoch.append(self.classifier.predict(self.predictions_processed))
      epoch += 1

  def predict(self, predictions: list):
    """Given a list of weak learners predictions, it predicts the output using the MLP
    classifier trained beforehand.
    """
    predictions_concatenated = self.concatenate_predictions(predictions)

    return self.classifier.predict(predictions_concatenated)
  
  def full_predict(self, data: list):
    """Given a list of data, it predicts the output using the MLP classifier trained
    beforehand.
    """
    predictions = [weak_learner.predict(data) for weak_learner in self.weak_learners]
    predictions_concatenated = self.concatenate_predictions(predictions)

    return self.classifier.predict(predictions_concatenated)


  def save(self, name = ""):
    """Saves the model as binaries using pickle native library so the model
    trained can be used later on without the need of re-training it. 
    """
    filename = name
    
    if not filename:
      filename = "meta-learner.lnr"

    if not path.exists("out/"):
      mkdir("out/")

    # The 'out/' folder will be the default folder to save the models
    with open("out/" + filename, "wb") as file:
      pickle.dump(self, file)

  def save_classifier(self, name = ""):
    """Saves the model MLP classifier as binaries using pickle native library so the model
    trained can be used later on without the need of re-training it. 
    """
    filename = name
    
    if not filename:
      filename = "meta-learner.clf"

    if not path.exists("out/"):
      mkdir("out/")

    # The 'out/' folder will be the default folder to save the models
    with open("out/" + filename, "wb") as file:
      pickle.dump(self.classifier, file)
  
  @staticmethod
  def load(filename: str) -> 'MetaLearner':
    """Loads a model previously trained from a file using pickle native library
    """
    with open("out/" + filename, "rb") as file:
      return pickle.load(file)
    
  @staticmethod
  def load_classifier(filename: str) -> 'MLPClassifier':
    """Loads a model MLP classifier previously trained from a file using pickle native library
    """
    with open("out/" + filename, "rb") as file:
      return pickle.load(file)