#!/usr/bin/env python3

# this code will detect table locations, then feed that location to tabula
# to segment the table into a csv

import os.path
import sys

from tabula import read_pdf

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import column_detection
import data_loader
import lexical
import line_classifier
import line_trigram
import table_detection


def eval(pdf_path, page_number, tm, em, cm, lm):
    csvs = list()
    page = data_loader.page_from_pdf(pdf_path, page_number)
    table_areas = table_detection.eval(tm, em, cm, lm, page)
    for index, area in enumerate(table_areas):
        tabula_area = (
                area.top * 72.0/300.0,
                area.left * 72.0/300.0,
                area.bottom * 72.0/300.0,
                area.right * 72.0/300.0,)
        extracted_tables = read_pdf(pdf_path, pages=page.page_number, area=tabula_area)
        for t in extracted_tables:
            csvs.append([index, t.to_json()])
    return csvs

if __name__ == '__main__':
    transition_model = line_trigram.load()
    emission_model = line_classifier.load()
    column_model = column_detection.load()
    lexical_model = lexical.load()

    csvs = eval(sys.argv[1], int(sys.argv[2]), transition_model, emission_model, column_model, lexical_model)
    for c in csvs:
        print(c)
