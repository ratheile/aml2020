#!/bin/bash

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml env/env.yml)

# scp * $USERNAME@euler.ethz.ch:$AML_CLUSTER
# Copying to Euler but excluding pickle and joblib files (othwerwise takes forever)
# rsync -P -r env modules $PROJECT --exclude={'*.pkl','*joblib','.git'} $USERNAME@euler.ethz.ch:$AML_CLUSTER/.
# scp * $USERNAME@login.leonhard.ethz.ch:$AML_CLUSTER
# rsync -P -r env modules $PROJECT --exclude={'*.pkl','*joblib','.git'} $USERNAME@login.leonhard.ethz.ch:$AML_CLUSTER/.
echo "Syncing to cluster account $euler_username, folder: $euler_cluster_dir"
rsync -P -r * --exclude={'*.pkl','*joblib','.git'} $euler_username@$euler_cluster_url:$euler_cluster_dir/.