import argparse
from training import train_model

def main(parser: argparse.ArgumentParser):
  args = parser.parse_args()

  if (args.load and args.train):
    print("[ERROR] You can't load the model and train it at the same time")
    return

  if args.train:
    train_model(args.params_path, args.verbose)


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
    action='store_true'
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

  main(parser)