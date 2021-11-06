#!/usr/bin/env python3

import os.path
import fitz
from collections import namedtuple
from glob import glob

import pdf2image

import tabio.config
import tabio.csv_file

Page = namedtuple('Page', ['hash', 'page_number',
                  'csv_fname', 'label_fname', 'background_fname'])


def find_matching_pdfs(label_fname):
    dir_name = os.path.dirname(label_fname)
    (doc_hash, _) = os.path.splitext(os.path.basename(label_fname))[0].split('_')
    pdf_fname = os.path.join(dir_name, f"{doc_hash}.cksum.pdf")
    if os.path.exists(pdf_fname):
        return pdf_fname
    else:
        print(f"Could not find cksum pdf for {doc_hash}")
    return None


def all_pages():
    label_files = glob(f'{tabio.config.in_dir}/*/*.xml')
    pdf_files = map(find_matching_pdfs, label_files)
    try:
        for label_fname, pdf in zip(label_files, pdf_files):
            doc = fitz.open(pdf)
            for pageNum in range(1, doc.page_count+1):
                yield page_from_pdf(pdf, pageNum, label_fname)
    except Exception as e:
        print(e)


def training_pages():
    for page in all_pages():
        if page.hash not in tabio.config.test_hashes:
            yield page


def test_pages():
    for page in all_pages():
        if page.hash in tabio.config.test_hashes:
            yield page


def page_from_pdf(pdf_path, page_number, label_fname=None):
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
        tabio.csv_file.create_csv_from_pdf(pdf_path, page_number),
        label_fname,
        background_fname)
    return page
