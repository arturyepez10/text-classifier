import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from os import path, mkdir
import pickle

from models.learner import Learner
from models.parameters.learner import Classifier

class WeakLearner(Learner):
  def __init__(self, model_name: str, classifier_params: Classifier):
    self.transformer = SentenceTransformer(model_name)
    self.model_name = model_name
    super().__init__(classifier_params)

    self.embeddings = np.array([])

  def generate_embeddings(self, data):
    return self.transformer.encode(data)
  
  def train_transformer(self, dataframe: pd.DataFrame):
    self.embeddings = self.generate_embeddings(dataframe["text"].to_list())
    self.classifier.fit(self.embeddings, dataframe["target"].to_list())

  def predict(self, data: list | None = None):
    embeddings = self.embeddings if not data else self.generate_embeddings(data)

    return self.classifier.predict(embeddings)
  
  def save(self, name = ""):
    """Saves the model as binaries using pickle native library so the model
    trained can be used later on without the need of re-training it. 
    """
    filename = name
    
    if not filename:
      filename = "meta-learner.clf"

    if not path.exists("out/"):
      mkdir("out/")

    # The 'out/' folder will be the default folder to save the models
    with open("out/" + filename, "wb") as file:
      pickle.dump(self, file)
  
  @staticmethod
  def load(filename: str) -> 'WeakLearner':
    """Loads a model previously trained from a file using pickle native library
    """
    with open("out/" + filename, "rb") as file:
      return pickle.load(file)