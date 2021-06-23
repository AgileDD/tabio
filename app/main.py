import ujson
import os
import traceback
import glob
from fastapi.responses import RedirectResponse
from tabio import line_trigram, line_classifier, column_detection, lexical, data_loader, table_detection, table_extraction, config
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from app.tabio_engine import TabioEngine
from app.util import safe_join

app = FastAPI(title="Tabio API",
              description="REST API endpoint for running documents through Tabio", redoc_url=None)
app.tabio_engine: TabioEngine = None


@app.on_event("startup")
async def startup():
    app.tabio_engine = TabioEngine(os.path.join("app", "models", "iqc_tabio"))


@app.post("/table_detect/")
async def table_detect(page: int, file: UploadFile = File(...)):
    """
        Returns the detected tables coords with table index.
    """
    try:
        contents = await file.read()
        with open(file.filename, "wb") as _temp:
            _temp.write(contents)
        locations = app.tabio_engine.detect(file.filename, page)
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=502, detail="Tabio failed with error {}".format(e))
    else:
        # this is file clean up    
        [os.remove(x) for x in glob.glob(
            "{}/{}*".format(os.path.abspath(os.getcwd()), ".".join(os.path.split(file.filename)[:-1])))]
        return ujson.dumps(locations)


@app.post("/table_extract/")
async def table_extract(page: int, file: UploadFile = File(...)):
    """
       Returns the detected tables from a page as json with table index.
    """
    csvs = None
    try:
        contents = await file.read()
        with open(file.filename, "wb") as _temp:
            _temp.write(contents)
        csvs = app.tabio_engine.inference(file.filename, page)
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=502, detail="Tabio failed with error {}".format(e))
    else:
        [os.remove(x) for x in glob.glob(
            "{}/{}*".format(os.path.abspath(os.getcwd()), ".".join(os.path.split(file.filename))[:-1]))]
        return ujson.dumps(csvs)


@app.post("/training/")
def training(dataset_dir: str, background_tasks: BackgroundTasks):
    """
        Train tabio models
    """
    path = safe_join("/data", os.path.join("/data", dataset_dir))
    background_tasks.add_task(app.tabio_engine.train, path)
    return {"result": "Started Training"}
