#!/opt/anaconda3/bin/python

import sys
import os.path
from pascalvoc import BBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import numpy as np
import statistics
from collections import namedtuple
import subprocess
import pascalvoc
import math


def bbox_union(a, b):
    if a == None:
        return b
    if b == None:
        return a
    return BBox(
        min(a.left, b.left), max(a.bottom, b.bottom),
        max(a.right, b.right), min(a.top, b.top))


#holds text, a list of bounding boxes for each char, and an overall box for the line
Line = namedtuple('Line', ['text', 'bboxes', 'bbox', 'side'])

#returns a list of Line structures
# multiple lines may be physically next to eachother on the page
# each line is separated by a '\n' char in the csv
def read_csv(fname):
    obj = csvfile2Object(fname)
    if len(obj) == 0:
        print("{} empty csv".format(fname))
        return []

    text = getText(obj)
    lines = []
    line_bbox = None
    line_bboxes = []
    line_chars = []

    for c, i in zip(text[1], text[2]):
        if c == '\n':
            lines.append(Line(''.join(line_chars), line_bboxes, line_bbox, 'unknown'))
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
            if line_bbox is not None:
                line = Line(''.join(line_chars), line_bboxes, line_bbox, 'unknown')
                spacial_lines.append(line)
            line_bboxes = [b]
            line_chars = [c]
            line_bbox = b
            line_top = b.top

    if line_bbox is not None:
        line = Line(''.join(line_chars), line_bboxes, line_bbox, 'unknown')
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
            if bbox.left < 0 or bbox.top < 0:
                continue

            min_x = min(min_x, bbox.left)
            min_y = min(min_y, bbox.top)
    return lines

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
                bottom=line.bbox.bottom-min_y),
            side='unknown'))

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


def create_csv_from_pdf(pdf_fname, page_number):
    '''
    This function gets the CSV from the pdf file and then converts into list of objects.
    '''
    pdfbox_jar = os.path.join(os.path.dirname(__file__), './ExtractText.jar')

    dir_name = os.path.dirname(pdf_fname)
    base_name, _ = os.path.splitext(os.path.basename(pdf_fname))
    out_path = os.path.join(dir_name, base_name+'_'+str(page_number)+'.csv')

    result = subprocess.run(
        [
            'java',
            "-jar",
            "-Xms1024m",
            "-Xmx8196m",
            pdfbox_jar,
            pdf_fname,
            str(page_number),
            out_path
        ],
        check=True,
        timeout=120)

    return out_path


class CSV2Obj():
    """CSV2Obj
    Converts a string read from a CSV format to an object.
    """

    def __init__(self, single_char):
            char_attributes = single_char.split(",")
            if len(char_attributes) == 17 or len(char_attributes) == 18:
                #ipdb.set_trace()
                #print("###################################3")
                #print(char_attributes)
                if char_attributes[0] == '"':
                    self.unicode = ','
                    #print("yo ",char_attributes[0])
                    #print(char_attributes)
                    char_attributes.remove(char_attributes[1])
                else:
                    #print(char_attributes[0])
                    self.unicode = char_attributes[0][1:-1]
                self.characterCode = int((char_attributes[1]))
                self.x = float(char_attributes[2])
                #print(char_attributes)
                self.y = float(char_attributes[3])
                self.fontSize = float(char_attributes[4])
                #print(char_attributes[5])
                self.fontSizeInPt = float(char_attributes[5])
                if len(char_attributes) == 17:
                    self.font = char_attributes[6].replace('"', '')

                elif len(char_attributes) == 18:
                    self.font = char_attributes[6].replace('"', '') + char_attributes[7].replace('"', '')
                    char_attributes.remove(char_attributes[7])
                elif len(char_attributes) == 19:
                    self.font = char_attributes[6].replace('"', '') + char_attributes[7].replace('"', '') + char_attributes[8].replace('"','')
                    char_attributes.remove(char_attributes[7])
                    char_attributes.remove(char_attributes[8])
                self.width = float(char_attributes[7])
                self.height = float(char_attributes[8])
                self.widthDirAdj = float(char_attributes[9])
                self.widthOfSpace = float(char_attributes[10])
                self.heightDir = float(char_attributes[11])
                self.xScale = float(char_attributes[12])
                self.yScale = float(char_attributes[13])
                self.dir = float(char_attributes[14])
                self.xDirAdj = float(char_attributes[15])
                self.yDirAdj = float(char_attributes[16])
            
            elif len(char_attributes) == 16 or len(char_attributes) == 15:
                if char_attributes[0] == '"':
                    self.unicode = ','
                    # print(char_attributes)
                    char_attributes.remove(char_attributes[1])
                else:
                    self.unicode = char_attributes[0].replace('"', '')
                self.characterCode = int((char_attributes[1]))
                self.x = float(char_attributes[2])
                #print(char_attributes)
                self.y = float(char_attributes[3])
                self.fontSize = float(char_attributes[4])
                #print(char_attributes[5])
                self.fontSizeInPt = float(char_attributes[5])
                if len(char_attributes) == 15:
                    self.font = char_attributes[6].replace('"', '')

                elif len(char_attributes) == 16:
                    self.font = char_attributes[6].replace('"', '') + char_attributes[7].replace('"', '')
                    char_attributes.remove(char_attributes[7])

                self.width = float(char_attributes[7])
                self.height = float(char_attributes[8])
                self.widthDirAdj = float(char_attributes[9])
                self.widthOfSpace = float(char_attributes[10])
                self.heightDir = float(char_attributes[11])
                self.xScale = float(char_attributes[12])
                self.yScale = float(char_attributes[13])
                self.dir = float(char_attributes[14])
                self.xDirAdj = float(char_attributes[2])
                self.yDirAdj =  float(char_attributes[3])
            else:
                raise Exception("invalid number of columns in csv '"+str(len(char_attributes))+"'")


def csvfile2Object(file_name):
    '''
    This function reads a csv file and converts into objects
    '''
    lines = []
    with open(file_name) as f:
        for line in f:
            lines.append(line)
    filetext = open(file_name,"r").read()
    #reading csv files
    lines= filetext.split("\n")
    # spliting csv file into lines
    a = len(lines)
    del lines[a-1]
    list_object = []
    for strings in lines:
        #print(strings)
        myobject = CSV2Obj(strings)
        list_object.append(myobject)
        #converting into objects and adding them to the list
    return list_object


def getText(structure):
    '''
    this function extracts rawText, FormattedText and mapping of character by taking in structure as in input.
    '''
    IndexList = []
    plainStringList = []
    FormattedStringList = []
    if len(structure)==0:
        return [[" "],[" "],[0]] # Nothing means an empty page or just a space
    xback = int(round(structure[0].xDirAdj))
    #getting the x co-ordinate of the first object
    yback = int(round(structure[0].yDirAdj))
    #getting the y co-ordinate of the first object
    width = int(math.floor(structure[0].widthDirAdj))
    avg_width = 0
    #getting the width of the first object
    for i in range(len(structure)):
          avg_width += structure[i].widthDirAdj
    index = 0
    avg_width = avg_width/len(structure)
    if avg_width == 0:
        avg_width = 1
        messages.Log("WARNING: Found average width zero in getText. Length of structure = "+str(len(structure)))

    for character in structure:
        # processing each object in list of objects
        plainStringList.append(character.unicode)
        # ipdb.set_trace()

        xco = int(round(character.xDirAdj))
        yco = int(round(character.yDirAdj))

        if width == 0:
            width = 1
        # setting to width to 1 if it comes to 0 after rounding off in order to avoid errors

        t_test = int(math.ceil(abs(xco-xback)/avg_width))
        # calculating the spacing between characters

        y_test = abs(yco-yback)
        # calculating the change of line

        if t_test >= 6 or y_test > 5:

            if y_test > 5:
                FormattedStringList.append("\n")
                IndexList.append("None")
                # if there is change in line , then linefeed is inserted
                FormattedStringList.append(character.unicode)
                IndexList.append(int(index))
                index += 1
            else:
                FormattedStringList.append(" ")
                # if there is some spacing, then space is inserted
                IndexList.append("None")

                FormattedStringList.append(character.unicode)
                IndexList.append(int(index))
                index += 1
        else:

            # plainStringList.append(character.unicode)
            FormattedStringList.append(character.unicode)
            IndexList.append(int(index))
            index += 1
        xback = xco
        yback = yco
        width = int((character.widthDirAdj))
    function_output = [plainStringList, FormattedStringList, IndexList]
    # print(plainStringList)
    #print(Format)
    return function_output


if __name__ == '__main__':
    doc_hash = '2ec6960ba691c6dc7880cabeafce2129'
    doc_csv_hash = '2ec6960ba691c6dc7880cabeafce2129'
    #doc_csv_hash = doc_hash

    #csv_contents = sql2box.charDictExtract(None, doc_hash, None)
    #for page,csv in csv_contents.items():
    #    with open(doc_hash+'_'+str(page)+'_'+str(page)+'.csv', 'wt') as f:
    #        f.write(csv)
    #sys.exit(0)

    page_number = int(sys.argv[1])
    base_fname = '../labeled-data/SortedIFP/'+doc_hash+'/'+doc_hash+'_'+str(page_number)+'_'+str(page_number)
    base_csv_fname = '../labeled-data/SortedIFP/'+doc_hash+'/'+doc_csv_hash+'_'+str(page_number)+'_'+str(page_number)




    labeled_boxes = pascalvoc.read(base_fname+'.xml')

    line_bboxes = []

    line_bbox = None
    line_chars = []
    line_category = None


    im = np.array(Image.open(base_fname+'.jpg'), dtype=np.uint8)

    fig, ax = plt.subplots(1)

    ax.imshow(im)

    lines = read_csv(base_csv_fname+'.csv')
    #lines = group_lines_spacially(lines)

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



