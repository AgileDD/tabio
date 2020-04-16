#!/usr/bin/env python

import sys
import os.path
sys.path.append('../IQC_Classification')

import textExtract

from pascalvoc import BBox

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import statistics
from collections import namedtuple

import pascalvoc



#import messages
#messages.Initialize(1)
#import tagdocs
import sql2box

def bbox_union(a, b):
    if a == None:
        return b
    if b == None:
        return a
    return BBox(
        min(a.left, b.left), max(a.bottom, b.bottom),
        max(a.right, b.right), min(a.top, b.top))


#holds text, a list of bounding boxes for each char, and an overall box for the line
Line = namedtuple('Line', ['text', 'bboxes', 'bbox'])

#returns a list of Line structures
# multiple lines may be physically next to eachother on the page
# each line is separated by a '\n' char in the csv
def read_csv(fname):
    obj = textExtract.csvfile2Object(fname)
    text = textExtract.getText(obj)

    lines = []

    line_bbox = None
    line_bboxes = []
    line_chars = []

    for c, i in zip(text[1], text[2]):
        if c == '\n':
            lines.append(Line(''.join(line_chars), line_bboxes, line_bbox))
            line_bbox = None
            line_bboxes = []
            line_chars = []
            continue

        bbox = None
        if i != 'None':
            info = obj[i]
            xScale = 300.0/72.0
            yScale = xScale
            bbox = BBox(
                (info.x)*xScale,
                (info.y)*yScale,
                (info.x+info.width)*xScale,
                (info.y-info.height)*yScale)

        line_bbox = bbox_union(line_bbox, bbox)
        line_chars.append(c)
        line_bboxes.append(bbox)

    return lines

# returns a list of Line structures
# each Line contains chars that share the same Y (+- epsilon)
# order of the input chars and lines is ignored
def group_lines_spacially(lines):
    if len(lines) == 0:
        return []

    size = mean_char_size(lines)
    y_separation = size[1] * .8

    # ignore the lines in the csv
    # find lines based on spacial location
    chars = []
    for line in lines:
        for c, b in zip(line.text, line.bboxes):
            if b is not None:
                chars.append((c, b))

    chars = sorted(chars, key=lambda x: x[1].top)

    spacial_lines = []

    line_chars = []
    line_bboxes = []
    line_top = chars[0][1].top
    line_bbox = None

    for c, b in chars:
        area = (b.bottom - b.top) * (b.right - b.left)
        #print(area)
        if area == 0.0:
            continue

        if b.top < (line_top + y_separation):
            line_chars.append(c)
            line_bboxes.append(b)
            line_bbox = bbox_union(line_bbox, b)
        else:
            line = Line(''.join(line_chars), line_bboxes, line_bbox)
            spacial_lines.append(line)
            line_bboxes = [b]
            line_chars = [c]
            line_bbox = b
            line_top = b.top

    if line_bbox is not None:
        line = Line(''.join(line_chars), line_bboxes, line_bbox)
        spacial_lines.append(line)
    return spacial_lines


def is_bbox_inside(outside, inside):
    return (outside.left < inside.left
        and outside.right > inside.right
        and outside.bottom < inside.bottom
        and outside.top > inside.top)

def find(f, seq):
    """Return first item in sequence where f(item) == True."""
    for item in seq:
      if f(item):
        return item
    return None

def mean_char_size(lines):
    char_widths = []
    char_heights = []

    for line in lines:
        for b in line.bboxes:
            if b is not None:
                char_widths.append(b.right - b.left)
                char_heights.append(b.bottom - b.top)

    return (statistics.mean(char_widths), statistics.mean(char_heights))

# detects any margins on the top and left side of the chars
# shifts all the chars (left, and up) to remove the margin
def remove_margin(lines):
    min_x = float("inf")
    min_y = float("inf")

    for line in lines:
        for bbox in line.bboxes:
            min_x = min(min_x, bbox.left)
            min_y = min(min_y, bbox.top)

    result = []
    for line in lines:
        shifted_bboxes = []
        for bbox in line.bboxes:
            shifted_bboxes.append(BBox(
                left=bbox.left-min_x,
                top=bbox.top-min_y,
                right=bbox.right-min_x,
                bottom=bbox.bottom-min_y))

        result.append(Line(
            text=line.text,
            bboxes=shifted_bboxes,
            bbox=BBox(
                left=line.bbox.left-min_x,
                top=line.bbox.top-min_y,
                right=line.bbox.right-min_x,
                bottom=line.bbox.bottom-min_y)))

    return result

def size(lines):
    x = 0.0
    y = 0.0
    for line in lines:
        for bbox in line.bboxes:
            x = max(x, bbox.right)
            y = min(y, bbox.bottom)
    return (x, y)

def draw(lines, ax):
    for line in lines:

        for char, bbox in zip(line.text, line.bboxes):
            if bbox is None:
                continue

            #shade over the char in blue
            b = bbox
            rect = patches.Rectangle(
                (b.left, b.bottom),
                (b.right-b.left),
                (b.top-b.bottom))
            ax.add_patch(rect)

        #draw red rectangle around whole line
        b = line.bbox
        rect = patches.Rectangle(
            (b.left, b.bottom),
            (b.right-b.left),
            (b.top-b.bottom),
            linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    #draw arrows between lines to show order
    for a, b in zip(lines, lines[1:]):
        dx = b.bbox.left - a.bbox.left
        dy = b.bbox.top - a.bbox.bottom
        ax.arrow(a.bbox.left, a.bbox.bottom, dx, dy, width=3, shape='full', length_includes_head=True)


# downloads all pages for a given document hash
def download(doc_hash, out_dir):
    csv_contents = sql2box.charDictExtract(None, doc_hash, None)
    for page,csv in csv_contents.items():
        print(page)
        fname = os.path.join(out_dir, doc_hash, doc_hash+'_'+str(page)+'_'+str(page)+'.csv')
        with open(fname, 'wt') as f:
            f.write(csv)

if __name__ == '__main__':
    doc_hash = '99e91ba482e98894f5c8da17dde5b259'
    doc_csv_hash = 'baad73b47f22e16520e04b02455c5817'
    #doc_csv_hash = doc_hash

    #csv_contents = sql2box.charDictExtract(None, doc_hash, None)
    #for page,csv in csv_contents.items():
    #    with open(doc_hash+'_'+str(page)+'_'+str(page)+'.csv', 'wt') as f:
    #        f.write(csv)
    #sys.exit(0)

    page_number = int(sys.argv[1])
    base_fname = '../labeled-data/Tarfaya/'+doc_hash+'_'+str(page_number)+'_'+str(page_number)
    base_csv_fname = '../labeled-data/Tarfaya/'+doc_csv_hash+'_'+str(page_number)+'_'+str(page_number)




    labeled_boxes = pascalvoc.read(base_fname+'.xml')

    line_bboxes = []

    line_bbox = None
    line_chars = []
    line_category = None


    im = np.array(Image.open(base_fname+'.jpg'), dtype=np.uint8)

    fig, ax = plt.subplots(1)

    ax.imshow(im)

    lines = read_csv(base_csv_fname+'.csv')

    for line in lines:

        line_category = None

        for char, bbox in zip(line.text, line.bboxes):
            if bbox is None:
                continue

            #detect line category
            if line_category is None:
                found_category = find(lambda x: is_bbox_inside(x.bbox, bbox), labeled_boxes)
                if found_category is not None:
                    line_category = found_category.name

        print(f"{line_category}\t\t{line.text}")



    draw(lines, ax)
    pascalvoc.draw(labeled_boxes, ax)

    plt.show()
    fig.savefig('fig.jpg', bbox_inches='tight', dpi=150)

