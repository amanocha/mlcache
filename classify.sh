#!/bin/bash

m=0
m_max=3
d_max=250

while [ $m -le $m_max ]
do
  echo $m

  d=250
  while [ $d -le $d_max ]
  do
    echo $d
    python3 classify.py --distance $d --model_num $m
    ((d = d+1))
  done

  ((m = m+1))
done

echo "Done!"
