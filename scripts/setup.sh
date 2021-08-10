#!/bin/bash
sudo apt-get install python3-venv python3-dev libglib2.0-dev libgirepository1.0-dev libcairo2-dev poppler-utils
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
