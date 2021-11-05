import os
import traceback

from app.exceptions import InvalidModelConfiguration
from tabio import line_trigram, line_classifier, column_detection, lexical, data_loader, table_detection, \
    table_extraction, config, train


def convert_to_json(data):
    output = list()
    for res in data:
        output_dict = {"top": res[0][0], "left": res[0][1], "bottom": res[0][2], "right": res[0][3], "table_data": []}
        for t in res[1]:
            output_dict["table_data"].append(t.to_csv())
        output.append(output_dict)
    return output


class TabioEngine:
    def __init__(self, model_path) -> None:
        self.model_path = model_path
        self.transition_model = None
        self.emission_model = None
        self.column_model = None
        self.lexical_model = None

    def load(self):
        # load the models
        self.transition_model = line_trigram.load(self.model_path)
        self.emission_model = line_classifier.load(self.model_path)
        self.column_model = column_detection.load(self.model_path)
        self.lexical_model = lexical.load(self.model_path)

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
        return convert_to_json(table_extraction.eval(
            pdf_path, page, self.transition_model, self.emission_model, self.column_model, self.lexical_model))

    def detect(self, pdf_path: str, page: int) -> list:
        """
            Detect tables for a specific page in a document
            Input:
                pdf_path: path of the pdf
                page: page number
            Return:
                list of cords and their index
                [{top, left, bottom, right}]
        """
        locations = list()
        page_data = data_loader.page_from_pdf(pdf_path, page)
        table_areas = table_detection.eval(
            self.transition_model, self.emission_model, self.column_model, self.lexical_model, page_data)
        for index, area in enumerate(table_areas):
            k = 72.0 / 300.0
            locations.append(
                {"top": area.top * k, "left": area.left * k, "bottom": area.bottom * k, "right": area.right * k})
        return locations

    def train(self, training_dir: str) -> None:
        """
            Tabio Training
            Input:
                data: path of dir with training data
            Return:
                None
        """
        try:
            print(f"Training on {training_dir}")
            config.in_dir = training_dir
            train.train(self.model_path)
            print(f"Training finished wrote to {self.model_path}")
        except Exception as e:
            print(f"Tabio training failed with error {e}\t{traceback.format_exc()}")
