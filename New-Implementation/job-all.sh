#!/bin/bash

for i in *.sh;
do

checking="job-all.sh"

if [ "$i" != "$checking" ]
then
  sbatch $i
fi

done
