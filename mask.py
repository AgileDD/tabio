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

    #resize the mask so each pixel represents about 1 character
    char_width, char_height = csv_file.mean_char_size(lines)
    x_scale = 1.0 / char_width
    y_scale = 1.0 / char_height
    new_width = (int)(mask.width * x_scale)
    new_height = (int)(mask.height * y_scale)

    mask = mask.resize((new_width, new_height), resample=Image.BICUBIC)
    #mask = Image.fromarray(np.uint8(decimate(mask, int(char_width) * 255)))
    return (mask, (x_scale, y_scale))


# split the mask into a mask image per line
# each line mask will include context above and below the line
def split(mask, scale, lines):
    mean_line_height = statistics.mean(
        map(lambda line: line.bbox.bottom - line.bbox.top, lines))
    width = mask.width

    masks = []

    for line in lines:
        top = line.bbox.top - (mean_line_height * 5.0)
        bottom = line.bbox.bottom + (mean_line_height * 5.0)
        box = (
            0,
            max(0, int(top * scale[1])),
            width,
            min(mask.height, int(bottom * scale[1])))

        masks.append(mask.crop(box))

    return masks


if __name__ == '__main__':
    fname = sys.argv[1]
    lines = csv_file.read_csv(fname)
    mask, scale = create(lines)
    mask.save('mask.png', 'PNG')

    masks = split(mask, scale, lines)
    for i, m in enumerate(masks):
        #print(i)
        m.save('mask-'+str(i)+'.png', 'PNG')
