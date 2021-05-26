import ujson
import os
import sys
import traceback
import glob
from fastapi.responses import RedirectResponse
from tabio import line_trigram, line_classifier, column_detection, lexical, data_loader, table_detection, table_extraction
from fastapi import FastAPI, HTTPException, File, UploadFile

app = FastAPI(title="Tabio API", description="REST API endpoint for running documents through Tabio", redoc_url=None)
transition_model = line_trigram.load()
emission_model = line_classifier.load()
column_model = column_detection.load()
lexical_model = lexical.load()

@app.post("/table_detect/")
async def table_detect(page: int, file: UploadFile = File(...)):
    """
        Returns the detected tables coords with table index.
    """
    _locations = list()
    try:
        contents = await file.read()
        with open(file.filename, "wb") as _temp:
            _temp.write(contents)
        page_data = data_loader.page_from_pdf(file.filename, page)
        table_areas = table_detection.eval(transition_model, emission_model, column_model, lexical_model, page_data)
        for index, area in enumerate(table_areas):
            k = 72.0/300.0
            _locations.append([index, (area.top*k, area.left*k, area.bottom*k, area.right*k)])
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(status_code=400, detail="Tabio failed with error {}".format(e))
    [os.remove(x) for x in glob.glob("{}/{}*".format(os.path.abspath(os.getcwd()), ".".join(file.filename.split('.')[:-1])))]
    return ujson.dumps(_locations)

@app.post("/table_extract/")
async def table_extract(page: int, file: UploadFile = File(...)):
    """
       Returns the detected tables from a page as json with table index.
    """
    try:
        contents = await file.read()
        with open(file.filename, "wb") as _temp:
            _temp.write(contents)
        csvs = table_extraction.eval(file.filename, page, transition_model, emission_model, column_model, lexical_model)
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(status_code=400, detail="Tabio failed with error {}".format(e))
    [os.remove(x) for x in glob.glob("{}/{}*".format(os.path.abspath(os.getcwd()), ".".join(file.filename.split('.')[:-1])))]
    return csvs

@app.post("/train/")
async def train():
    """
        Train custom tabio models
    """
    pass
