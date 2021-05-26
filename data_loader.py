#!/usr/bin/env python3

import os.path
from collections import namedtuple
from glob import glob
import sys

import pdf2image

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import config
import csv_file

Page = namedtuple('Page', ['hash', 'page_number', 'csv_fname', 'label_fname', 'background_fname'])

in_dir = config.in_dir
test_hashes = config.test_hashes

def find_matching_file(label_fname, extension):
    simple_fname = os.path.splitext(label_fname)[0]+extension
    if os.path.exists(simple_fname):
        return simple_fname

    # part of the page deosn't match. Label files have two pages numbers but only 1 matters
    dir_name = os.path.dirname(label_fname)
    (doc_hash, page_number) = os.path.splitext(os.path.basename(label_fname))[0].split('_')

    single_page_fname = os.path.join(dir_name, doc_hash+'_'+str(page_number)+extension)
    if os.path.exists(single_page_fname):
        return single_page_fname

    double_page_fname = os.path.join(dir_name, doc_hash+'_'+str(page_number)+'_'+str(page_number)+extension)
    if os.path.exists(double_page_fname):
        return double_page_fname

    return None


def find_matching_csv(label_fname):
    return find_matching_file(label_fname, '.csv')

def find_matching_jpg(label_fname):
    return find_matching_file(label_fname, '.jpg')


def all_pages():
    label_files = glob(f'{in_dir}/*/*.xml')
    csv_files = map(find_matching_csv, label_files)
    background_files = map(find_matching_jpg, label_files)

    for (label_fname, csv_fname, background_fname) in zip(label_files, csv_files, background_files):
        if csv_fname is None:
            continue

        (doc_hash, page_number) = os.path.splitext(os.path.basename(csv_fname))[0].split('_')

        yield Page(doc_hash, page_number, csv_fname, label_fname, background_fname)


def training_pages():
    for page in all_pages():
        if page.hash not in test_hashes:
            yield page


def test_pages():
    for page in all_pages():
        if page.hash in test_hashes:
            yield page


def page_from_pdf_data(pdf_name, pdf_path, page_number):
    dirname, pdf_name = os.path.split(pdf_name)
    hash = os.path.splitext(pdf_name)[0]
    background_fname = os.path.join(dirname, hash+'_'+str(page_number)+'.jpg')
    if not os.path.exists(background_fname):
        image = pdf2image.convert_from_bytes(
            open(pdf_path, 'rb').read(),
            first_page=page_number,
            last_page=page_number,
            dpi=300)
        image[0].save(background_fname, 'JPEG')
    page = Page(
        hash,
        page_number,
        csv_file.create_csv_from_pdf(pdf_path, page_number),
        None,
        background_fname)
    return page    

# returns a page data structure given a real pdf and page number
# - pdf does not have to be in the data directory
def page_from_pdf(pdf_path, page_number):
    dirname, pdf_name = os.path.split(pdf_path)

    hash = os.path.splitext(pdf_name)[0]

    background_fname = os.path.join(dirname, hash+'_'+str(page_number)+'.jpg')
    if not os.path.exists(background_fname):
        image = pdf2image.convert_from_bytes(
            open(pdf_path, 'rb').read(),
            first_page=page_number,
            last_page=page_number,
            dpi=300)
        image[0].save(background_fname, 'JPEG')

    page = Page(
        hash,
        page_number,
        csv_file.create_csv_from_pdf(pdf_path, page_number),
        None,
        background_fname)

    return page
