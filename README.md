# TabIO

Extraction of tables from documents using machine learning and graphs

How to use:

Download and extract SortedIFP.zip

1) download csv pdfbox files form iqc_dev

`./get_csv_files.sh ./SortedIFP`


2) edit data_loader.py set `in_dir` path

3) train a column decector
`python3 train_column_detector.py --train`

4) train a line trigram model

`python3 line_trigram.py`


5) trail a line classifier

`python3 line_classifier.py --train`


6) evaluate

`python3 viterbi.py`