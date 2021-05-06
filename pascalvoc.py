import xml.etree.ElementTree as et
from collections import namedtuple
import matplotlib.patches as patches

BBox = namedtuple('BBox', ['left', 'bottom', 'right', 'top'])
NamedBox = namedtuple('NamedBox', ['name', 'bbox'])

def read(fname):
    doc = et.parse(fname)
    root = doc.getroot()

    box_names = []



    for o in root.iter('object'):
        name = o.find('name').text

        bbox = o.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymin = int(float(bbox.find('ymin').text))
        ymax = int(float(bbox.find('ymax').text))

        box_names.append(NamedBox(name, bbox=BBox(xmin, ymin, xmax, ymax)))

    return box_names

def draw(labeled_boxes, ax):
    for box in labeled_boxes:
        b = box.bbox
        rect = patches.Rectangle(
            (b.left, b.bottom),
            (b.right-b.left),
            (b.top-b.bottom),
            linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)