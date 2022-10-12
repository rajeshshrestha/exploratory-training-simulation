#!/bin/bash
gunicorn -w "$(sysctl -n hw.physicalcpu)" --bind 0.0.0.0:5000 --log-level info --timeout 240 api:app