#!/bin/sh

python -m pytest --cov=scriptchat --cov-report=term-missing --cov-report=html
