#!/bin/bash
for i in *.sh;
do
    if "$i" != "job-all.sh"
    then 
        echo "$i"
    fi
done