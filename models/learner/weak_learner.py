import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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
  
  # TODO: Define how we want to store the model with pickle 
  # def save(self, filename = ""):
  #   with open(filename, 'wb') as file:
  #     pickle.dump(self, file)