#!/usr/bin/env python

import sys
sys.path.append('../IQC_Classification')

import textExtract

from pascalvoc import BBox

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import sys
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
        min(a.left, b.left), min(a.bottom, b.bottom),
        max(a.right, b.right), max(a.top, b.top))


#holds text, a list of bounding boxes for each char, and an overall box for the line
Line = namedtuple('Line', ['text', 'bboxes', 'bbox'])

#returns a list of Line structures
# multiple lines may be physically next to eachother on the page
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
                (info.y-info.height)*xScale,
                (info.x+info.width)*xScale,
                (info.y)*yScale)

        line_bbox = bbox_union(line_bbox, bbox)
        line_chars.append(c)
        line_bboxes.append(bbox)

    return lines


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
                char_heights.append(b.top - b.bottom)

    return (statistics.mean(char_widths), statistics.mean(char_heights))


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

            #shade over the char in blue
            b = bbox
            rect = patches.Rectangle(
                (b.left, b.bottom),
                (b.right-b.left),
                (b.top-b.bottom))
            ax.add_patch(rect)

        print(f"{line_category}\t\t{line.text}")

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
        dy = b.bbox.bottom - a.bbox.top
        ax.arrow(a.bbox.left, a.bbox.top, dx, dy, width=3, shape='full', length_includes_head=True)

    #draw labeled boxes
    for box in labeled_boxes:

        b = box.bbox
        rect = patches.Rectangle(
            (b.left, b.bottom),
            (b.right-b.left),
            (b.top-b.bottom),
            linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)

    plt.show()
    fig.savefig('fig.jpg', bbox_inches='tight', dpi=150)

