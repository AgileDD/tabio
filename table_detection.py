#!/usr/bin/env python3

# this code will detect table locations on a page
#
# table detection uses viterbi search to classify lines. Adjacent lines that 
# are classified as a table will be grouped together

import sys
import data_loader
import line_trigram
import frontend
import pascalvoc
import column_detection
import table_detection
import line_classifier
import viterbi
import csv_file
import torch
import os.path
import itertools
import functools



# given a list of lines and a classification for each line, return a list of
# bounding boxes. Each bounding box will represent the area of a table on the page
def detect_tables(lines, classifications):
    
    is_table = map(lambda c: 'Table' in c or 'Frame' in c, classifications)

    # find consecutive lines that represent tables, ensuring the table does not
    # cross a boundary where the number of columns change
    # (A table will never be both on the left column and center column)
    tables = []
    for key, group in itertools.groupby(zip(lines, is_table), key=lambda i: (i[0].side, i[1])):
        if key[1]:
            tables.append([i[0].bbox for i in group])

    return map(lambda t: functools.reduce(csv_file.bbox_union, t), tables)

def detect_left_columns(lines):
    # find consecutive lines that represent tables, ensuring the table does not
    # cross a boundary where the number of columns change
    # (A table will never be both on the left column and center column)
    areas = []
    for key, group in itertools.groupby(lines, key=lambda l: l.side):
        if key == 'left':
            areas.append([i.bbox for i in group])

    return map(lambda t: functools.reduce(csv_file.bbox_union, t), areas)

#given all our trained models, and a page, returns a list of rectangles representing areas of tables on the page
def eval(transition_model, emission_model, column_model, page):
    features, lines = frontend.create(page, lambda ls, ms: column_detection.eval(column_model, ms))
    hypothesis = viterbi.search_page(transition_model, emission_model, features)
    table_areas = table_detection.detect_tables(lines, hypothesis)
    #table_areas = detect_left_columns(lines)
    return table_areas


if __name__ == '__main__':
    transition_model = line_trigram.load()
    emission_model = line_classifier.load()
    column_model = column_detection.load()

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    page = data_loader.page_from_pdf(pdf_path, page_number)

    table_areas = eval(transition_model, emission_model, column_model, page)

    for area in table_areas:
        k = 72.0/300.0
        print(f'{area.top*k} {area.left*k} {area.bottom*k} {area.right*k}')