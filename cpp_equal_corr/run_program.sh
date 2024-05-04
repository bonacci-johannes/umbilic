#!/bin/bash
num=100
gamma=0.25
length=100000
t_max=5000

# Loop 30 times
for ((i=33; i<=50; i++))
do
    # Echo which job is being started
    echo "Starting job $i with arguments: $num $gamma $length $t_max"
    # Call the program in the background
    ./program $i $num $gamma $length $t_max &
done
