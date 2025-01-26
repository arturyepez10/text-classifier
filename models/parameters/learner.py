import numpy as np

from models.parameters.classifier import Classifier

class Learner:
  def __init__(self, classifier):
    self.classifier = Classifier(
      hidden_layer_sizes=classifier["hidden_layer_sizes"],
      random_state=classifier["random_state"],
      early_stopping=classifier["early_stopping"]
    )

  def __str__(self):
    return f"Classifier: \n{self.classifier}"

class WeakLeaner(Learner):
  def __init__(self, models, classifier):
    self.models = np.array(models)
    super().__init__(classifier)

  def __str__(self):
    return f"Models: {self.models}\n\t{super().__str__()}"

class MetaLearner(Learner):
  def __init__(self, classifier):
    super().__init__(classifier)

  def __str__(self):
    return super().__str__()