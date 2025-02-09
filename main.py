import argparse
from models.parameters import Parameters
from training import train_model
from loading import load_model
from predictions import predict_file
from metrics import score_model

from models.learner.meta_learner import MetaLearner

def main(parser: argparse.ArgumentParser):
  args = parser.parse_args()

  # Load the parameters from the file path passed as argument to the execution
  params = Parameters.from_json(args.params_path)

  if (args.load and args.train):
    print("[ERROR] You can't load the model and train it at the same time")
    return

  meta_learner : MetaLearner = None

  if args.train:
    meta_learner = train_model(params, args.verbose)

  if args.load:
    meta_learner = load_model(args.load, args.verbose)

  if (args.predict or args.score) and not meta_learner:
    print("[ERROR] You need to train or load the model first")
    return

  if args.predict:
    predict_file(args.predict, meta_learner, args.verbose)

  if args.score:
    score_model(params, meta_learner, args.verbose)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Text Classifier for Emotion Detection | SENTI-Lib")

  parser.add_argument(
    "-v",
    "--verbose",
    help="Print several logs during the whole process",
    action='store_true'
  )

  parser.add_argument(
    "-l",
    "--load",
    help="Load the model from previous training",
    action='store'
  )

  parser.add_argument(
    "-t",
    "--train",
    help="Train the model",
    action='store_true'
  )

  parser.add_argument(
    "-pr",
    "--predict",
    help="Given a path, predicts the emotion of the text in the file",
    action='store'
  )

  parser.add_argument(
    "-p",
    "--params-path",
    help="Path to the parameters file to train the classifier",
    action='store',
    default="parameters.default.json"
  )

  parser.add_argument(
    "-s",
    "--score",
    help="After a model is loaded, it obtain several metrics from the model",
    action='store_true'
  )

  main(parser)