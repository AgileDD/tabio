from csv_file import read_csv, mean_char_size, Line, bbox_union
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np

doc_hash = '99e91ba482e98894f5c8da17dde5b259'
doc_csv_hash = 'baad73b47f22e16520e04b02455c5817'

page_number = int(sys.argv[1])
base_fname = '../labeled-data/Tarfaya/'+doc_hash+'_'+str(page_number)+'_'+str(page_number)
base_csv_fname = '../labeled-data/Tarfaya/'+doc_csv_hash+'_'+str(page_number)+'_'+str(page_number)


lines = read_csv(base_csv_fname+'.csv')
size = mean_char_size(lines)
y_separation = size[1] * .8
print(y_separation)

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
print(line_top)
for c, b in chars:
    if b.top < (line_top + y_separation):
        line_chars.append(c)
        line_bboxes.append(b)
        line_bbox = bbox_union(line_bbox, b)
    else:
        print(b.top, line_top)
        line = Line(''.join(line_chars), line_bboxes, line_bbox)
        spacial_lines.append(line)
        line_bboxes = [b]
        line_chars = [c]
        line_bbox = b
        line_top = b.top
        

line = Line(''.join(line_chars), line_bboxes, line_bbox)
spacial_lines.append(line)


im = np.array(Image.open(base_fname+'.jpg'), dtype=np.uint8)
fig, ax = plt.subplots(1)
ax.imshow(im)

for line in spacial_lines:

        line_category = None

        for char, bbox in zip(line.text, line.bboxes):
            if bbox is None:
                continue

            #detect line category
            #if line_category is None:
            #    found_category = find(lambda x: is_bbox_inside(x.bbox, bbox), labeled_boxes)
            #    if found_category is not None:
            #        line_category = found_category.name

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


plt.show()
