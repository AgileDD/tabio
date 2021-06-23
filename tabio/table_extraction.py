#!/usr/bin/env python3

# this code will detect table locations, then feed that location to tabula
# to segment the table into a csv

import os.path
import sys

from tabula import read_pdf

import tabio.column_detection
import tabio.data_loader
import tabio.lexical
import tabio.line_classifier
import tabio.line_trigram
import tabio.table_detection


def eval(pdf_path, page_number, tm, em, cm, lm):
    page = tabio.data_loader.page_from_pdf(pdf_path, page_number)
    table_areas = tabio.table_detection.eval(tm, em, cm, lm, page)
    for index, area in enumerate(table_areas):
        tabula_area = (
            area.top * 72.0/300.0,
            area.left * 72.0/300.0,
            area.bottom * 72.0/300.0,
            area.right * 72.0/300.0,)
        extracted_tables = read_pdf(
            pdf_path, pages=page.page_number, area=tabula_area)
        outputs = [(tabula_area, [])]  # [(cords, [index, table data])]
        for t in extracted_tables:
            outputs[0][1].append((index, t.to_json()))
    return outputs


if __name__ == '__main__':
    transition_model = tabio.line_trigram.load(os.path.join("app", "models", "iqc_tabio"))
    emission_model = tabio.line_classifier.load(os.path.join("app", "models", "iqc_tabio"))
    column_model = tabio.column_detection.load(os.path.join("app", "models", "iqc_tabio"))
    lexical_model = tabio.lexical.load(os.path.join("app", "models", "iqc_tabio"))

    output = eval(sys.argv[1], int(sys.argv[2]), transition_model,
                emission_model, column_model, lexical_model)
    cords, tables = output[0]
    print(cords)
    for t in tables:
        print(t)
