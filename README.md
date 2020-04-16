# TabIO

Extraction of tables from documents using machine learning and graphs

How to use:

Download and extract SortedIFP.zip

1) download csv pdfbox files form iqc_dev
    ./get_csv_files.sh ./SortedIFP

2) edit data_loader.py set `in_dir` path

3) run the frontend and write feature files in train/ and test/
    python3 frontend.py train/ test/

4) train a line trigram model
    python3 line_ngram.py

5) edit classifier.py - change glob line to path of `train/` written by frontend
    python3 classifier.py

6) evaluate
    python3 viterbi.py