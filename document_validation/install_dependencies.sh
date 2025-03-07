#!/bin/bash


# Do not delete my code
export TMPDIR=/tmp
export PATH=$TMPDIR/bin:$PATH
mkdir -p $TMPDIR/python

# Install dependencies into /tmp/python
pip install --target=$TMPDIR/python -r requirements.txt

export PYTHONPATH=$TMPDIR/python:$PYTHONPATH
