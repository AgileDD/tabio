#!/usr/bin/env python

# used to create a mask image for a document that represents
# where the characters are located
#
# used for creating features when classifying sections of a document

import sys
import csv_file
from PIL import Image, ImageDraw

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
    new_width = (int)(1.0 * mask.width / char_width)
    new_height = (int)(1.0 * mask.height / char_height)

    mask = mask.resize((new_width, new_height), resample=Image.BICUBIC)
    return mask

if __name__ == '__main__':
    fname = sys.argv[1]
    lines = csv_file.read_csv(fname)
    mask = create(lines)
    mask.save('mask.png', 'PNG')
