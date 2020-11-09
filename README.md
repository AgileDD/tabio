# TabIO

Extraction of tables from documents using machine learning and graphs

## How to use:

1) Run setup.sh to install dependencies

2) Train or download model files

3) Call `table_detection.py <pdf_path> <page_number>` to detect table areas on a page 


## Training

To train models, edit config.py and set `in_dir` to the location of your training data. The training scripts expect a folder structure containing jpg images of the pages of your pdfs, and xml label files describing the structure of the page. The xml files are in pascalvoc format.

The expected folder structure is:
in_dir/<document_hash>/<document_hash>_<page_number>.jpg
in_dir/<document_hash>/<document_hash>_<page_number>.xml

There should be a folder per document, and in each document folder there should be an image of each page, and an associated label xml file for each page in the document. In `config.py` you can configure a test set, which consists of a lsit of document hashes. These documents will be excluded from training and can be used to evaluate the models.