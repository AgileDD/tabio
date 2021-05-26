import tabio.config
import tabio.column_detection
import tabio.line_classifier

if __name__ == "__main__":
    if tabio.config.enable_column_detection:
        tabio.column_detection.train()
    tabio.line_classifier.train()
