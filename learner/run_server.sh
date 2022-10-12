#!/bin/bash
python3.7 -m gunicorn -w "$(nproc)" --bind 0.0.0.0:5000 --log-level info --timeout 240 api:app