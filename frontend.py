#!/usr/bin/env python3
import os
import sys
import errno
import csv_file
import pascalvoc
import mask
import split_lines
import data_loader
import column_detection
from PIL import Image


def read_lines(page):
    lines = csv_file.read_csv(page.csv_fname)
    lines = csv_file.group_lines_spacially(lines)
    lines = csv_file.remove_margin(lines)
    return lines


def read_labels(page):
    return pascalvoc.read(page.label_fname)


# first stange of feature generation
# generates a feature vector per line
# does not split up columns
#
# feature generation is split up into stages because the output
# of stage 1 is needed to train a column classifier
def stage1(lines):
    doc_mask = mask.create(lines)
    masks = mask.split(doc_mask, lines)

    return masks

# calls the column classifier on each line
# then double column features are split so the output
# list of features can be treated as though the whole
# page is single column
def stage2(lines, masks, column_classifier):
    columns = list(map(column_classifier, lines))
    # columns = eval_col.column_detector(masks)
    feature_vectors = split_lines.split_masks(masks, columns)
    feature_vectors = map(lambda m: m.resize((100, 20), resample=Image.BICUBIC), feature_vectors)

    lines = split_lines.split_lines(lines, columns)

    #after splitting lines, some parts may be None due to having no characters
    # in one half after splitting. Remove the corresponding feature vector
    # since it represents blank space
    feature_vectors_final = []
    lines_final = []
    for f, l in zip(feature_vectors, lines):
        if l is not None:
            feature_vectors_final.append(f)
            lines_final.append(l)

    return (feature_vectors_final, lines_final)


# create feature vectors for the lines in a page
def create(page, column_classifier):
    lines = read_lines(page)
    masks = stage1(lines)

    #now we have 1 mask per line, detect columns
    return stage2(lines, masks, column_classifier)


if __name__ == '__main__':
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]

    def mkdir(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    mkdir(test_dir)
    mkdir(train_dir)

    # todo: implement real column detection, for now
    #       get the info from the labeled boxes
    

    for page in data_loader.all_pages():
        print(page)

        labeled_boxes = read_labels(page)
        

        (features, lines) = create(page, lambda l: column_detection.fake_column_detection(l, labeled_boxes))
        labels = map(lambda l: column_detection.read_line_classification(l, labeled_boxes), lines)

        out_dir = test_dir if page.hash in data_loader.test_hashes else train_dir
        for i, (feature, label) in enumerate(zip(features, labels)):
            if label is None:
                continue
            label = label.split('-')[1]
            mkdir(f"{out_dir}/{label}")
            feature.save(f"{out_dir}/{label}/{page.hash}_{page.page_number}_{i}.png", 'PNG')
