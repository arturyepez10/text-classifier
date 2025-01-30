import argparse
from training import train_model

def main(parser: argparse.ArgumentParser):
  args = parser.parse_args()

  train_model(args.p, args.v)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Text Classifier for Emotion Detection | SENTI-Lib")

  parser.add_argument(
    "-verbose",
    "--v",
    help="Print several logs during the whole process",
    action='store_true'
  )

  parser.add_argument(
    "-params-path",
    "--p",
    help="Path to the parameters file to train the classifier",
    action='store',
    required=True
  )

  main(parser)