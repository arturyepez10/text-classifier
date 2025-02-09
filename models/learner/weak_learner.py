import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier

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
    """Embeddings are genereated using the instance Sentence Transformer
    used to create the weak learner.

    [WARNING]\n
    In order to optimize the disk space when model is stored, the transformer
    is generated and then removed from the instance to save only the embeddings
    generated.

    If you find yourself offline and without the local instance of the transformer
    models to use, there may be errors when trying to generate the embeddings.
    """

    if not self.transformer:
      self.transformer = SentenceTransformer(self.model_name)

    embeddings = self.transformer.encode(data)

    self.transformer = None

    return embeddings
  
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
      filename = self.model_name + ".lnr"

    # Optimize the size of binaries by removing the SentenceTransformer
    self.transformer = None

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
      filename = self.model_name + ".clf"

    if not path.exists("out/"):
      mkdir("out/")

    # The 'out/' folder will be the default folder to save the models
    with open("out/" + filename, "wb") as file:
      pickle.dump(self.classifier, file)
  
  @staticmethod
  def load(filename: str) -> 'WeakLearner':
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