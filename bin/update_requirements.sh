#!/bin/sh
python -m pip freeze --exclude-editable > requirements.txt
