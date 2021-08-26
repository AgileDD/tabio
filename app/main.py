import glob
import os
import tempfile
import traceback
from uuid import UUID

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks

from app.tabio_engine import TabioEngine
from app.util import safe_join
from . import jobs


app = FastAPI(title="Tabio API",
              description="REST API endpoint for running documents through Tabio", redoc_url=None)
jobs.connect()


def run_job(uid, func, *args):
    job = jobs.find(uid)
    try:
        job.result = func(*args)
        job.status = "complete"
    except Exception as e:
        print(f"Job {uid} failed with error {e}\t{traceback.format_exc()}")
        job.status = "failed"
    finally:
        job.save()


def start_job(background_tasks, func, *args):
    new_task = jobs.Job()
    new_task.save()
    background_tasks.add_task(run_job, new_task.uid, func, *args)
    return new_task


@app.post("/table_detect/")
async def table_detect(page: int, file: UploadFile = File(...)):
    """
        Returns the detected tables cords with table index.
    """
    tabio = TabioEngine(os.path.join("/app",  "tabio", "models", "iqc_tabio"))
    try:
        contents = await file.read()
        junk_file_name = ""
        with tempfile.NamedTemporaryFile() as temp:
            junk_file_name = temp.name
            temp.write(contents)
            tabio.load()
            return tabio.detect(temp.name, page)
    except Exception as e:
        print("Failure with tabio: {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=500, detail="Tabio failed with error {}".format(e))
    finally:
        [os.remove(x) for x in glob.glob("{}*".format(junk_file_name))]


@app.post("/table_extract/")
async def table_extract(page: int, file: UploadFile = File(...)):
    """
       Returns the detected tables from a page as json with table index.
    """
    tabio = TabioEngine(os.path.join("/app", "tabio", "models", "iqc_tabio"))
    try:
        contents = await file.read()
        junk_file_name = ""
        with tempfile.NamedTemporaryFile() as temp:
            junk_file_name = temp.name
            temp.write(contents)
            tabio.load()
            return tabio.inference(temp.name, page)
    except Exception as e:
        print("Failure with tabio: {}\n{}".format(e, traceback.format_exc()))
        raise HTTPException(
            status_code=500, detail="Tabio failed with error {}".format(e))
    finally:
        [os.remove(x) for x in glob.glob("{}*".format(junk_file_name))]


@app.post("/train/")
def train(dataset_dir: str, background_tasks: BackgroundTasks):
    """
        Train tabio models
    """
    tabio = TabioEngine(os.path.join("/app", "tabio", "models", "iqc_tabio"))
    path = safe_join("/data", os.path.join("/data", "tabio_training_data", dataset_dir))
    return start_job(background_tasks, tabio.train, path)


@app.get("/task/{uid}/status")
def training_status(uid: UUID):
    """
        Training status
    """
    return jobs.find(uid)
