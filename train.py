import config
import column_detection
import line_classifier

if config.enable_column_detection:
    column_detection.train()
line_classifier.train()
