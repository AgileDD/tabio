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


def find_matching_file(label_fname, extension):
    simple_fname = os.path.splitext(label_fname)[0]+extension
    if os.path.exists(simple_fname):
        return simple_fname

    # part of the page deosn't match. Label files have two pages numbers but only 1 matters
    dir_name = os.path.dirname(label_fname)
    (doc_hash, page_number) = os.path.splitext(
        os.path.basename(label_fname))[0].split('_')

    # for pdfs we only want cksum
    pdf_fname = os.path.join(dir_name, f"{doc_hash}.cksum.pdf")
    if os.path.exists(pdf_fname):
        return pdf_fname

    single_page_fname = os.path.join(
        dir_name, doc_hash+'_'+str(page_number)+extension)
    if os.path.exists(single_page_fname):
        return single_page_fname

    double_page_fname = os.path.join(
        dir_name, doc_hash+'_'+str(page_number)+'_'+str(page_number)+extension)
    if os.path.exists(double_page_fname):
        return double_page_fname

    return None


def find_matching_pdfs(label_fname):
    return find_matching_file(label_fname, '.cksum.pdf')


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


def test_hashes():
    return None