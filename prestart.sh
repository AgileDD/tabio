#!/bin/bash

# start the jobs server process which holds the list of jobs
# for all the gunicorn processes
python3 /app/app/job.py&
