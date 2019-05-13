#!/bin/bash

t=1

max=$((2**20))

while [ $t -le $max ]
do
  echo $t
  python3 preprocess.py --mode divide --num_clusters $t
  ((t = t*2))
done

echo "Done!"
