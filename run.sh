#!/bin/bash

counter=1

for notebook in models/*.ipynb; do
    echo "Running notebook $counter: $notebook"
    jupyter nbconvert --to python --execute "$notebook"
    ((counter++))
done