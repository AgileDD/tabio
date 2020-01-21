from csv_file import read_csv, mean_char_size, Line, bbox_union, group_lines_spacially, is_bbox_inside
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import pascalvoc

doc_hash = '99e91ba482e98894f5c8da17dde5b259'
doc_csv_hash = 'baad73b47f22e16520e04b02455c5817'

page_number = int(sys.argv[1])
base_fname = '../labeled-data/Tarfaya/'+doc_hash+'_'+str(page_number)+'_'+str(page_number)
base_csv_fname = '../labeled-data/Tarfaya/'+doc_csv_hash+'_'+str(page_number)+'_'+str(page_number)


lines = read_csv(base_csv_fname+'.csv')
spacial_lines = group_lines_spacially(lines)
labeled_boxes = pascalvoc.read(base_fname+'.xml')

def classify_line(line):
    for bbox in line.bboxes:
        for l in labeled_boxes:
            if is_bbox_inside(l.bbox, bbox):
                return l.name
    return None


im = np.array(Image.open(base_fname+'.jpg'), dtype=np.uint8)
fig, ax = plt.subplots(1)
ax.imshow(im)

for line in spacial_lines:

        line_category = classify_line(line)
        print(line_category)

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


        #print(f"{line_category}\t\t{line.text}")

        #draw red rectangle around whole line
        b = line.bbox
        if b is None:
            continue
        rect = patches.Rectangle(
            (b.left, b.bottom),
            (b.right-b.left),
            (b.top-b.bottom),
            linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

pascalvoc.draw(labeled_boxes, ax)

plt.show()
