#!/bin/bash
# Run from the Fit_bite project root (the folder that contains app/)
cd "$(dirname "$0")"
python -m pip install -q -r requirements-web.txt
python -m uvicorn server:app --reload
