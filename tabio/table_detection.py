#!/usr/bin/env python3

# this code will detect table locations on a page
#
# table detection uses viterbi search to classify lines. Adjacent lines that
# are classified as a table will be grouped together

import functools
import itertools
import os.path
import sys

import torch

import tabio.column_detection
import tabio.csv_file
import tabio.data_loader
import tabio.frontend
import tabio.lexical
import tabio.line_classifier
import tabio.line_trigram
import tabio.pascalvoc
import tabio.table_detection
import tabio.viterbi

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


# given a list of lines and a classification for each line, return a list of
# bounding boxes. Each bounding box will represent the area of a table on the page
def detect_tables(lines, classifications):

    print(f"{classifications=}")
    is_table = map(lambda c: 'Table' in c or 'Frame' in c, classifications)

    # find consecutive lines that represent tables, ensuring the table does not
    # cross a boundary where the number of columns change
    # (A table will never be both on the left column and center column)
    tables = []
    for key, group in itertools.groupby(zip(lines, is_table), key=lambda i: (i[0].side, i[1])):
        if key[1]:
            tables.append([i[0].bbox for i in group])

    return map(lambda t: functools.reduce(tabio.csv_file.bbox_union, t), tables)


def detect_left_columns(lines):
    # find consecutive lines that represent tables, ensuring the table does not
    # cross a boundary where the number of columns change
    # (A table will never be both on the left column and center column)
    areas = []
    for key, group in itertools.groupby(lines, key=lambda l: l.side):
        if key == 'left':
            areas.append([i.bbox for i in group])

    return map(lambda t: functools.reduce(tabio.csv_file.bbox_union, t), areas)

# given all our trained models, and a page, returns a list of rectangles representing areas of tables on the page


def eval(transition_model, emission_model, column_model, lexical_model, page):
    features, lines = tabio.frontend.create(
        page, lambda ls, ms: tabio.column_detection.eval(column_model, ms))
    lexical_features = tabio.lexical.create_lexical_features(
        lexical_model, lines)
    hypothesis = tabio.viterbi.search_page(
        transition_model, emission_model, features, lexical_features)
    table_areas = tabio.table_detection.detect_tables(lines, hypothesis)
    #table_areas = detect_left_columns(lines)
    return table_areas


if __name__ == '__main__':
    transition_model = tabio.line_trigram.load(os.path.join("/app", "tabio", "models", "iqc_tabio"))
    emission_model = tabio.line_classifier.load(os.path.join("/app","tabio",  "models", "iqc_tabio"))
    column_model = tabio.column_detection.load(os.path.join("/app","tabio",  "models", "iqc_tabio"))
    lexical_model = tabio.lexical.load(os.path.join("/app","tabio",  "models", "iqc_tabio"))

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])

    page = tabio.data_loader.page_from_pdf(pdf_path, page_number)

    table_areas = eval(transition_model, emission_model,
                       column_model, lexical_model, page)

    for area in table_areas:
        k = 72.0/300.0
        print(f'{area.top*k} {area.left*k} {area.bottom*k} {area.right*k}')
