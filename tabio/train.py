import tabio.column_detection
import tabio.config
import tabio.line_classifier
import tabio.lexical
import tabio.line_trigram
import sys


def train(model_path):
    if tabio.config.enable_column_detection:
        tabio.column_detection.train(model_path)
    tabio.lexical.train(model_path)
    tabio.line_classifier.train(model_path)
    tabio.line_trigram.train(model_path)


if __name__ == "__main__":
    train(sys.argv[1])
