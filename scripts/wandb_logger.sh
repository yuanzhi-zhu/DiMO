#!/bin/bash

result_dir="outputs/wandb"
# Get all the subdirectories containing latest-run

ref_dir_list=($(ls -d ${result_dir}/*/*/wandb/latest-run 2>/dev/null))
echo "Found directories: ${ref_dir_list[@]}"

while true; do
    ref_dir_list=($(ls -d ${result_dir}/*/*/wandb/latest-run 2>/dev/null))
    for DIR in "${ref_dir_list[@]}"; do
        echo "Syncing: $DIR"
        wandb sync "$DIR"
    done
    sleep 300
done
