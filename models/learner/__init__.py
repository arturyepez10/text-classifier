from sklearn.neural_network import MLPClassifier

from models.parameters.classifier import Classifier

class Learner:
  def __init__(self, params: Classifier):
    self.classifier = MLPClassifier(
      hidden_layer_sizes=params.hidden_layer_sizes,
      # TODO: Add support for the following parameters (both class and this constructor)
      # activation=params.activation,
      # solver=params.solver,
      # alpha=params.alpha,
      random_state=params.random_state,
      early_stopping=params.early_stopping
    )