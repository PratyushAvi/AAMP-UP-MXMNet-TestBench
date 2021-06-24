#!/bin/bash

scales=(0.001 0.025 0.05 0.10 0.15 0.20 0.25)
runtime=60

for scale in "${scales[@]}"
do
    python3 benchmarking.py --scale="$scale" --runtime="$runtime"
    echo -e "\nrun with $scale of QM9 complete \n"
done
