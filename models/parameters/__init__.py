import json
from os.path import isfile

from models.parameters.learner import WeakLeaner, MetaLearner
from models.parameters.dataset import Dataset

class Parameters:

  def __init__(self, weak_learner, meta_learner, dataset):
    self.weak_learner = WeakLeaner(
      models=weak_learner["models"],
      classifier=weak_learner["classifier"]
    )
    self.meta_learner = MetaLearner(
      classifier=meta_learner["classifier"]
    )
    self.dataset = Dataset(
      train=dataset["train"],
      test=dataset["test"],
      file_type=dataset["file_type"],
      text_column=dataset["text_column"],
      label_columns=dataset["label_columns"],
      sep=dataset["separator"]
    )
    
  @staticmethod
  def from_json(path: str):
    if not isfile(path):
      raise FileNotFoundError(f"[ERROR] File '{path}' not found for loading Parameters.")
    
    with open(path) as file:
      parameters = json.load(file)

    return Parameters(parameters["weak_learner"], parameters["meta_learner"], parameters["dataset"])
  
  def __str__ (self):
    return f"Weak Learner: \n\t{self.weak_learner}\nMeta Learner: \n\t{self.meta_learner}\nDataset: \n\t{self.dataset}"