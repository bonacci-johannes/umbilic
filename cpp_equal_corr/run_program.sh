#!/bin/bash

# Define variables
num=50
length=100000
t_max=10000
total_jobs=128  # Total number of jobs to start
max_concurrent_jobs=64  # Maximum number of concurrent jobs
dest_path="gam_series"  # Path to save output files, choose '.' for current directory

# Function to ensure only $max_concurrent_jobs are running at the same time
function check_jobs() {
    while true; do
        # Count number of currently running jobs
        running_jobs=$(jobs -rp | wc -l)
        if [[ "$running_jobs" -lt "$max_concurrent_jobs" ]]; then
            break  # Exit loop if we can start new jobs
        fi
        sleep 1  # Wait a bit before checking again
    done
}

# Loop over gamma values from 0.1 to 1.0 in steps of 0.05
for gamma in $(seq 0.75 0.05 1.0)
# for gamma in 0.1
do
    # Loop to start all jobs for the current gamma
    for ((i=1; i<=total_jobs; i++))
    do
        check_jobs  # Call function to check number of running jobs
        # Echo which job is being started
        echo "Starting job $i for gamma $gamma with arguments: $num $gamma $length $t_max $dest_path"
        # Start the job in the background
        ./umbilic_two_dir_corr $i $num $gamma $length $t_max $dest_path &
    done
done

# Wait for all jobs to complete
wait
