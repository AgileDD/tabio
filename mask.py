#!/usr/bin/env python

# used to create a mask image for a document that represents
# where the characters are located
#
# used for creating features when classifying sections of a document

import sys
import csv_file
from PIL import Image, ImageDraw
import statistics
from scipy.signal import decimate
import numpy as np


def create(lines):
    max_x=0
    max_y=0
    for line in lines:
        for bbox in line.bboxes:
            if bbox is None:
                continue
            max_x = max(max_x, bbox.right)
            max_y = max(max_y, bbox.bottom)

    mask = Image.new("L", (int(max_x), int(max_y)))
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([(0,0),mask.size], fill=255)

    for line in lines:
        for bbox in line.bboxes:
            if bbox is None:
                continue
            mask_draw.rectangle([
                bbox.left,
                bbox.top,
                bbox.right,
                bbox.bottom], fill=0)

    return mask


# split the mask into a mask image per line
# each line mask will include context above and below the line
def split(mask, lines):
    mean_line_height = statistics.mean(
        map(lambda line: line.bbox.bottom - line.bbox.top, lines))
    width = mask.width

    masks = []

    for line in lines:
        top = line.bbox.top - (mean_line_height * 5.0)
        bottom = line.bbox.bottom + (mean_line_height * 5.0)
        box = (
            0,
            max(0, int(top)),
            width,
            min(mask.height, int(bottom)))

        masks.append(mask.crop(box))

    return masks


if __name__ == '__main__':
    fname = sys.argv[1]
    lines = csv_file.read_csv(fname)
    mask = create(lines)
    mask.save('mask.png', 'PNG')

    masks = split(mask, lines)
    for i, m in enumerate(masks):
        #print(i)
        m.save('mask-'+str(i)+'.png', 'PNG')
