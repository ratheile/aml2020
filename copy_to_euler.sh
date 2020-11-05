#!/bin/bash

USERNAME=inesb # Change this to your username
PROJECT=project2 # Change to project folder you want to copy
AML_EULER=aml2020

scp * $USERNAME@euler.ethz.ch:$AML_EULER
# Copying to Euler but excluding pickle and joblib files (othwerwise takes forever)
rsync -P -r env modules $PROJECT --exclude={'*.pkl','*joblib','.git'} $USERNAME@euler.ethz.ch:$AML_EULER/.