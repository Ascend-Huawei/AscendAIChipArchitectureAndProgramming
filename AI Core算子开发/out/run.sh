#!/bin/sh

if [ -d "output" ]; then
  rm -rf output
fi

mkdir output

./main -i input.txt -o output.txt -e expect.txt -b ../operator/kernel_meta/Reduction.o -p 0.8 -d 0.8 -k Reduction__kernel0 -t 0