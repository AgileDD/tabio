import csv_file

def read_line_classification(line, labeled_boxes):
    for bbox in line.bboxes:
        for l in labeled_boxes:
            if csv_file.is_bbox_inside(l.bbox, bbox):
                return l.name
    return None

def fake_column_detection(line, labeled_boxes):
    classification = read_line_classification(line, labeled_boxes)
    if classification is None:
        return None
    return classification.split('-')[0]