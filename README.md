# TabIO

Extraction of tables from documents using machine learning and graphs

## How to use:

1) Run setup.sh to install dependencies

2) Train or download model files

3) Call `table_detection.py <pdf_path> <page_number>` to detect table areas on a page 

4) Call `table_extraction.py <pdf_path> <page_number>` to extract table data


## Training

To train models, edit config.py and set `in_dir` to the location of your training data. The training scripts expect a folder structure containing jpg images of the pages of your pdfs, a CSV file containing PDFBOX output for each page, and LabelImg generated XML label files describing the structure of the page. The xml files are in pascalvoc format. See "## CSV File Generation" below on how to generate the CSV files. 

The expected folder structure is:
in_dir/<document_hash>/<document_hash>_<page_number>.jpg
in_dir/<document_hash>/<document_hash>_<page_number>.xml
in_dir/<document_hash>/<document_hash>_<page_number>.csv

Here, the <document_hash> refers to the md5sum of a document, although another unique prefix for each document would also work. 

In `config.py` you can configure a test set, which consists of a list of document hashes. These documents will be excluded from training and can be used to evaluate the models.

The following sequence of steps runs the training of the models:

python3 train.py  # Trains the column detector and the line classification models. This will train the models col_trained_net.pth and trained_net.pth. 
python3 line_trigram.py # Trains the prior probability model as a trigram. This will train the model line_ngram.pkl
python3 lexical.py # Trains the TFIDF-SVD modek and saves it to lexical_model.pickle. 

Run Viterbi decoding to get a text output:
python3 viterbi.py # Run an evaluation on the test hashes defined in the config file.

## CSV File Generation
To generate a PDFBOX CSV file for a page of a PDF file, run:
java -jar -Xms1024m -Xmx8196m ExtractText.jar <PDF_file> <page_number> <document_hash>_<page_number>.csv
This needs to be done for each page of the PDF for training. 
