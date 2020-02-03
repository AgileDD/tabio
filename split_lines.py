import sys
import os
import errno
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import pascalvoc
import mask
import csv_file

matplotlib.use('GTK3Cairo')

#doc_hash = '99e91ba482e98894f5c8da17dde5b259'
#doc_csv_hash = 'baad73b47f22e16520e04b02455c5817'

doc_hash = sys.argv[1]
page_number = int(sys.argv[2])
base_fname = '../labeled-data/IFP Samples/'+doc_hash+'_'+str(page_number)+'_'+str(page_number)
#base_csv_fname = '../labeled-data/Tarfaya/'+doc_csv_hash+'_'+str(page_number)+'_'+str(page_number)
base_csv_fname = base_fname


lines = csv_file.read_csv(base_csv_fname+'.csv')
spacial_lines = csv_file.remove_margin(csv_file.group_lines_spacially(lines))
labeled_boxes = pascalvoc.read(base_fname+'.xml')

doc_mask = mask.create(spacial_lines)
masks = mask.split(doc_mask, spacial_lines)

def classify_line(line):
    for bbox in line.bboxes:
        for l in labeled_boxes:
            if csv_file.is_bbox_inside(l.bbox, bbox):
                return l.name
    return None


# split the line into two parts where the first part contains chars.x < width
def split_line(line, width):
    left = None
    right = None
    def add(c, bbox, out_line):
        if out_line is None:
            return csv_file.Line(text=c, bboxes=[bbox], bbox=bbox)
        return csv_file.Line(
            text=out_line.text+c,
            bboxes=out_line.bboxes+[bbox],
            bbox=csv_file.bbox_union(out_line.bbox, bbox))

    for c, bbox in zip(line.text, line.bboxes):
        if bbox.right < width:
            left = add(c, bbox, left)
        else:
            right = add(c, bbox, right)

    return (left, right)



#im = np.array(Image.open(base_fname+'.jpg'), dtype=np.uint8)
#fig, ax = plt.subplots(1)
#ax.imshow(im)

(width, _) = csv_file.size(spacial_lines)

# split up lines, labels and masks further by column
# single column entries can be carried over as is, but
# double column entries will be split in half and both halfs added
# to the following lists
split_lines = []
split_labels = []
split_masks = []

for line, mask in zip(spacial_lines, masks):

        line_category = classify_line(line)
        if line_category is None:
            continue

        #print(line_category)

        column_type, label = line_category.split('-')

        if column_type == 'SingleColumn':
            split_lines.append(line)
            split_labels.append(label)
            split_masks.append(mask)
            print(label)
        if column_type == 'DoubleColumn':
            #split line and mask in half
            l,r = split_line(line, width/2.0)

            mask_cut_point = mask.width / 2
            lmask = mask.crop((0, 0, mask_cut_point, mask.height))
            rmask = mask.crop((mask_cut_point, 0, mask.width, mask.height))

            if l is not None:
                l_cat = classify_line(l).split('-')[1]
                if l_cat is not None:
                    split_lines.append(l)
                    split_labels.append(l_cat)
                    split_masks.append(lmask)
                    print('Left -'+split_labels[-1])
            if r is not None:
                r_cat = classify_line(r).split('-')[1]
                if r_cat is not None:
                    split_lines.append(r)
                    split_labels.append(r_cat)
                    split_masks.append(rmask)
                    print('Right-'+split_labels[-1])



#csv_file.draw(split_lines, ax)

#pascalvoc.draw(labeled_boxes, ax)

#plt.show()

def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

out_dir = 'sorted_masks'
mkdir(out_dir)

for i, (line, label, mask) in enumerate(zip(split_lines, split_labels, split_masks)):
    mkdir(f"{out_dir}/{label}")
    out_mask = mask.resize((100, 20), resample=Image.BICUBIC)
    out_mask.save(f"{out_dir}/{label}/{doc_hash}_{page_number}_{i}.png", 'PNG')
