#!/bin/bash
cd $HOME/aml2020 # moves you from your cluster home directory to the aml2020 folder

# Why you shouldn't parse ls output: https://mywiki.wooledge.org/ParsingLs
for file in experiments/test1/project2*; do # will need changing if we change the path or the naming of the yml files.
    bsub -n 4 -R "rusage[mem=10000]" ./run_project.sh $file # To understand the dollar sign + @: https://stackoverflow.com/questions/5163144/what-are-the-special-dollar-sign-shell-variables
done