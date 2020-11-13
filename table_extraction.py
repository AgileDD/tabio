#!/usr/bin/env python3

# this code will detect table locations, then feed that location to tabula
# to segment the table into a csv


import data_loader
import line_trigram
import column_detection
import table_detection
import line_classifier
from tabula import read_pdf
import os.path


if __name__ == '__main__':
    transition_model = line_trigram.load()
    emission_model = line_classifier.load()
    column_model = column_detection.load()

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    page = data_loader.page_from_pdf(pdf_path, page_number)

    table_areas = table_detection.eval(transition_model, emission_model, column_model, page)

    for area in table_areas:
        tabula_area = (
                area.top * 72.0/300.0,
                area.left * 72.0/300.0,
                area.bottom * 72.0/300.0,
                area.right * 72.0/300.0,
            )
        #gets a pandas dataframe
        extracted_table = read_pdf(pdf_path, pages=page.page_number, area=tabula_area)
        print(extracted_table)

    print('')
    print('')
