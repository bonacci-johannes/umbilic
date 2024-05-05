#!/bin/bash
num=50
gamma=0.25
length=100000
t_max=1000

# Loop 30 times
for ((i=1; i<=64; i++))
do
    # Echo which job is being started
    echo "Starting job $i with arguments: $num $gamma $length $t_max"
    # Call the program in the background
    ./program $i $num $gamma $length $t_max &
done
