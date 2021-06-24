import ujson
import os
import traceback
import glob
import tempfile
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from app.tabio_engine import TabioEngine
from app.util import safe_join

app = FastAPI(title="Tabio API",
              description="REST API endpoint for running documents through Tabio", redoc_url=None)
app.tabio_engine: TabioEngine = None


@app.on_event("startup")
async def startup():
    app.tabio_engine = TabioEngine(os.path.join("/app", "models", "iqc_tabio"))
    print("Tabio started with model {}".format(app.tabio_engine.model_path))


@app.post("/table_detect/")
async def table_detect(page: int, file: UploadFile = File(...)):
    """
        Returns the detected tables coords with table index.
    """
    try:
        contents = await file.read()
        junk_file_name = ""
        with tempfile.NamedTemporaryFile() as temp:
            junk_file_name = temp.name
            temp.write(contents)
            return ujson.dumps(app.tabio_engine.detect(temp.name, page))
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=500, detail="Tabio failed with error {}".format(e))
    finally:
        [os.remove(x) for x in glob.glob("{}*".format(junk_file_name))]


@app.post("/table_extract/")
async def table_extract(page: int, file: UploadFile = File(...)):
    """
       Returns the detected tables from a page as json with table index.
    """
    try:
        contents = await file.read()
        junk_file_name = ""
        with tempfile.NamedTemporaryFile() as temp:
            junk_file_name = temp.name
            temp.write(contents)
            return ujson.dumps(app.tabio_engine.inference(temp.name, page))
    except Exception as e:
        print("Failure with tabio {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=500, detail="Tabio failed with error {}".format(e))
    finally:
        [os.remove(x) for x in glob.glob("{}*".format(junk_file_name))]


@app.post("/training/")
def training(dataset_dir: str, background_tasks: BackgroundTasks):
    """
        Train tabio models
    """
    path = safe_join("/data", os.path.join("/data", dataset_dir))
    background_tasks.add_task(app.tabio_engine.train, path)
    return {"result": "Started Training"}
