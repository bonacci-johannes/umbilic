#!/bin/bash
num=50
gamma=0.25
length=100000
t_max=10000

# Loop 30 times
for ((i=1; i<=32; i++))
do
    # Echo which job is being started
    echo "Starting job $i with arguments: $num $gamma $length $t_max"
    # Call the program in the background
    ./umbilic_two_dir_corr $i $num $gamma $length $t_max &
done
