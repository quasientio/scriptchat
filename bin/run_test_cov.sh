#!/bin/sh

python -m pytest --cov=litechat --cov-report=term-missing --cov-report=html
