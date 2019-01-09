#!/bin/bash

echo "Testing medoids" && \
../exec/medoids -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10 && \
echo "Testing skmeans" && \
../exec/skmeans -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10 && \
echo "Testing fcm" && \
../exec/fcm -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10 #&& \
#echo "Testing mb_knori" && \
#exec/mb_knori -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 2 -i 10 -M 30
