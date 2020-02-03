import sys
import os
import errno
from glob import glob
import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import pascalvoc
import mask
import csv_file


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

#csv_file.draw(split_lines, ax)

#pascalvoc.draw(labeled_boxes, ax)

#plt.show()

def mkdir(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# creates line features for a given page in a document
def create_features(lines, labeled_boxes):
    doc_mask = mask.create(lines)
    masks = mask.split(doc_mask, lines)

    (width, _) = csv_file.size(lines)

    split_lines = []
    split_labels = []
    split_masks = []

    for line, whole_mask in zip(lines, masks):

            line_category = classify_line(line)
            if line_category is None:
                continue

            #print(line_category)

            column_type, label = line_category.split('-')

            if column_type == 'SingleColumn':
                split_lines.append(line)
                split_labels.append(label)
                split_masks.append(whole_mask)
                print(label)
            if column_type == 'DoubleColumn':
                #split line and mask in half
                l,r = split_line(line, width/2.0)

                mask_cut_point = whole_mask.width / 2
                lmask = whole_mask.crop((0, 0, mask_cut_point, whole_mask.height))
                rmask = whole_mask.crop((mask_cut_point, 0, whole_mask.width, whole_mask.height))

                if l is not None:
                    l_cat = classify_line(l)
                    if l_cat is not None:
                        l_cat = l_cat.split('-')[1]
                        split_lines.append(l)
                        split_labels.append(l_cat)
                        split_masks.append(lmask)
                        print('Left -'+split_labels[-1])
                if r is not None:
                    r_cat = classify_line(r)
                    if r_cat is not None:
                        r_cat = r_cat.split('-')[1]
                        split_lines.append(r)
                        split_labels.append(r_cat)
                        split_masks.append(rmask)
                        print('Right-'+split_labels[-1])

    features = map(lambda m: m.resize((100, 20), resample=Image.BICUBIC), split_masks)

    return (split_lines, features, split_labels)


def find_matching_csv(label_fname):
    simple_csv_fname = os.path.splitext(label_fname)[0]+'.csv'
    if os.path.exists(simple_csv_fname):
        return simple_csv_fname

    # part of the page deosn't match. Label files have two pages numbers but only 1 matters
    dir_name = os.path.dirname(label_fname)
    (doc_hash, _, page_number) = os.path.splitext(os.path.basename(label_fname))[0].split('_')

    single_page_fname = os.path.join(dir_name, doc_hash+'_'+str(page_number)+'.csv')
    if os.path.exists(single_page_fname):
        return single_page_fname

    double_page_fname = os.path.join(dir_name, doc_hash+'_'+str(page_number)+'_'+str(page_number)+'.csv')
    if os.path.exists(double_page_fname):
        return double_page_fname

    return None



if __name__ == '__main__':
    in_dir = sys.argv[1]
    train_dir = sys.argv[2]
    test_dir = sys.argv[3]
    test_hashes = set([line.strip() for line in open(sys.argv[4])])

    label_files = glob(f'{in_dir}/*.xml')
    csv_files = map(find_matching_csv, label_files)

    mkdir(test_dir)
    mkdir(train_dir)

    for (label_fname, csv_fname) in zip(label_files, csv_files):
        if csv_fname is None:
            continue

        (doc_hash, _, page_number) = os.path.splitext(os.path.basename(csv_fname))[0].split('_')
        print((doc_hash, page_number, label_fname, csv_fname))

        lines = csv_file.read_csv(csv_fname)
        lines = csv_file.remove_margin(csv_file.group_lines_spacially(lines))
        labeled_boxes = pascalvoc.read(label_fname)

        (_, features, labels) = create_features(lines, labeled_boxes)

        out_dir = test_dir if doc_hash in test_hashes else train_dir

        for i, (feature, label) in enumerate(zip(features, labels)):
            mkdir(f"{out_dir}/{label}")
            feature.save(f"{out_dir}/{label}/{doc_hash}_{page_number}_{i}.png", 'PNG')
