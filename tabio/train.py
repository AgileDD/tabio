import tabio.column_detection
import tabio.config
import tabio.line_classifier
import tabio.lexical
import tabio.line_trigram
import sys


def train(path):
    if tabio.config.enable_column_detection:
        tabio.column_detection.train(path)
    tabio.lexical.train(path)
    tabio.line_classifier.train(path)
    tabio.line_trigram.train(path)


if __name__ == "__main__":
    train(sys.argv[1])
