#!/bin/bash

../exec/medoids -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
../exec/skmeans -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
../exec/fcm -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 32 -i 10
../exec/mb_knori -f ../test-data/iris.bin -n 150 -m 4 -k 3 -T 2 -i 10 -M 10
