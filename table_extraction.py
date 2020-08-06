#!/usr/bin/env python3

# this code will detect table locations, then feed that location to tabula
# to segment the table into a csv


import data_loader
import line_trigram
import column_detection
import table_detection
import line_classifier

import torch
import itertools
import functools
from tabula import read_pdf
import os.path


if __name__ == '__main__':
    transition_model = line_trigram.load()
    emission_model = line_classifier.load()
    column_model = column_detection.load()

    all_hypothesis = []
    ground_truths = []
    all_lines = []
    for page in data_loader.test_pages():
        print(page.hash, page.page_number)
        table_areas = table_detection.eval(transition_model, emission_model, column_model, page)

        def pdf_fname(page):
            dirname = os.path.dirname(page.csv_fname)
            return os.path.join(dirname, page.hash+'.pdf')

        for area in table_areas:
            tabula_area = (
                    area.top * 72.0/300.0,
                    area.left * 72.0/300.0,
                    area.bottom * 72.0/300.0,
                    area.right * 72.0/300.0,
                )
            extracted_table = read_pdf(pdf_fname(page), pages=page.page_number, area=tabula_area)
            print(extracted_table)
        print('')
        print('')
