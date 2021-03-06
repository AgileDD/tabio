import errno
import os
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import tabio.csv_file
import tabio.data_loader
import tabio.mask
import tabio.pascalvoc


# split the line into two parts where the first part contains chars.x < width
def split_line(line, width):
    left = None
    right = None

    def add(c, bbox, out_line, side):
        if out_line is None:
            return tabio.csv_file.Line(text=c, bboxes=[bbox], bbox=bbox, side=side)
        return tabio.csv_file.Line(
            text=out_line.text+c,
            bboxes=out_line.bboxes+[bbox],
            bbox=tabio.csv_file.bbox_union(out_line.bbox, bbox),
            side=side)

    for c, bbox in zip(line.text, line.bboxes):
        if bbox.right < width:
            left = add(c, bbox, left, 'left')
        else:
            right = add(c, bbox, right, 'right')

    return (left, right)

# split  mask in half


def split_mask(mmask):
    mask_cut_point = mmask.width / 2
    lmask = mmask.crop((0, 0, mask_cut_point, mmask.height))
    rmask = mmask.crop((mask_cut_point, 0, mmask.width, mmask.height))
    return (lmask, rmask)

# given a list of items, and a list identifiying each item as 'SingleColumn' or 'DoubleColumn'
# split the items if they are a double column
#
# Each item will be split according to the item_splitter function
#
# The output order will follow a natural reading order


def split_and_order(items, columns, item_splitter):
    output = []
    current_left = []
    current_right = []
    status = 'double'

    for (item, column_type) in zip(items, columns):
        if column_type == 'SingleColumn':
            if status == 'double':
                output += current_left
                output += current_right
                current_left = []
                current_right = []
                status = 'single'
            output.append(item)
        if column_type == 'DoubleColumn':
            l, r = item_splitter(item)
            if l is not None:
                current_left.append(l)
            if r is not None:
                current_right.append(r)
            status = 'double'

    output += current_left
    output += current_right
    return output

# given a list of masks, some of which are double columns
# split only the double columnd lines, and return
# a singular list of all the masks
#
# this is usefull so all the masks can be treated
# individually and as a single column


def split_masks(masks, columns):
    return split_and_order(masks, columns, split_mask)


def split_lines(lines, columns):
    (width, _) = tabio.csv_file.size(lines)
    return split_and_order(lines, columns, lambda l: split_line(l, width / 2.0))
