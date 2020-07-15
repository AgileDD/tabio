#!/usr/bin/env python3

from collections import namedtuple
from glob import glob
import os.path
import config


Page = namedtuple('Page', ['hash', 'page_number', 'csv_fname', 'label_fname'])

in_dir = config.in_dir
test_hashes = config.test_hashes

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

def all_pages():
    label_files = glob(f'{in_dir}/*/*.xml')
    csv_files = map(find_matching_csv, label_files)

    for (label_fname, csv_fname) in zip(label_files, csv_files):
        if csv_fname is None:
            continue

        (doc_hash, _, page_number) = os.path.splitext(os.path.basename(csv_fname))[0].split('_')

        yield Page(doc_hash, page_number, csv_fname, label_fname)


def training_pages():
    for page in all_pages():
        if page.hash not in test_hashes:
            yield page


def test_pages():
    for page in all_pages():
        if page.hash in test_hashes:
            yield page

