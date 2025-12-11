#!/bin/bash


first_job=$1
last_job=$2

echo "Cancelling jobs from $first_job to $last_job..."

for job_id in $(seq "$first_job" "$last_job"); do
  echo "Cancelling job $job_id"
  scancel "$job_id"
done
