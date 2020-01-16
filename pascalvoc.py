import xml.etree.ElementTree as et
from collections import namedtuple

BBox = namedtuple('BoundingBox', ['left', 'bottom', 'right', 'top'])
NamedBox = namedtuple('NamedBox', ['name', 'bbox'])

def read(fname):
    doc = et.parse(fname)
    root = doc.getroot()

    box_names = []



    for o in root.iter('object'):
        name = o.find('name').text

        bbox = o.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)

        box_names.append(NamedBox(name, bbox=BBox(xmin, ymin, xmax, ymax)))

    return box_names