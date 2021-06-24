import os

from tabio import line_trigram, line_classifier, column_detection, lexical, data_loader, table_detection, table_extraction, config

from app.exceptions import ApplicationError, InvalidModelConfiguration

class TabioEngine():
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        # validate model config files
        self.validate_model()
        # load the models
        self.transition_model = line_trigram.load(model_path)
        self.emission_model = line_classifier.load(model_path)
        self.column_model = column_detection.load(model_path)
        self.lexical_model = lexical.load(model_path)

    def inference(self, pdf_path: str, page: int) -> list:
        """
            Get tables for a specific page in a document
            Input:
                pdf_path: path of the pdf
                page: page number
            Return:
                list of tables and their results
                [(cords, table data)]
        """
        return table_extraction.eval(
                pdf_path, page, self.transition_model, self.emission_model, self.column_model, self.lexical_model)

    def detect(self, pdf_path: str, page: int) -> list:
        """
            Detect tables for a specific page in a document
            Input:
                pdf_path: path of the pdf
                page: page number
            Return:
                list of cords and their index
                [index, (top, left, bottom, right)]
        """
        locations = list()  # [index, (top, left, bottom, right)]
        page_data = data_loader.page_from_pdf(pdf_path, page)
        table_areas = table_detection.eval(
            self.transition_model, self.emission_model, self.column_model, self.lexical_model, page_data)
        for index, area in enumerate(table_areas):
            k = 72.0/300.0
            locations.append(
                [index, (area.top*k, area.left*k, area.bottom*k, area.right*k)])
        return locations

    def train(self, training_dir: str) -> None:
        """
            Tabio Training
            Input:
                data: path of dir with training data
            Return:
                None
        """
        config.in_dir = training_dir
        if config.enable_column_detection:
            column_detection.train(self.model_path)
        line_classifier.train(self.model_path)

    def validate_model(self) -> bool:
        """
            File validation
            Return:
                True if model files exist
        """
        if not os.path.exists(os.path.join(self.model_path, "col_trained_net.pt")):
            raise InvalidModelConfiguration('col_trained_net.pt not found')
        if not os.path.exists(os.path.join(self.model_path, "lexical_model.pt")):
            raise InvalidModelConfiguration('lexical_model.pt not found')
        if not os.path.exists(os.path.join(self.model_path, 'line_ngram.pt')):
            raise InvalidModelConfiguration('line_ngram.pt not found')
        if not os.path.exists(os.path.join(self.model_path, 'trained_net.pt')):
            raise InvalidModelConfiguration('trained_net.pt not found')
        return True
