#!/bin/bash

m=0
m_max=5
d_max=0

while [ $m -le $m_max ]
do
  echo $m

  d=0
  while [ $d -le $d_max ]
  do
    echo $d
    python bandits_base.py --distance $d --model_num $m
    ((d = d+1))
  done

  ((m = m+1))
done

echo "Done!"
