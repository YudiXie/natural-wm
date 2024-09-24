#!/bin/bash
for jobid in {33702703..33702708}
do
    scancel $jobid
done