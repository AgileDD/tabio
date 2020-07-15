#!/bin/bash

# download corresponding csv files to match hand labels in xml files
# assumes csv and xml files are named like '<hash>_<page>_<page>.[csv|xml]'
set -e

target_dir=$1
pushd "$target_dir"
hashes=$(find -L -name '*.xml'|xargs -n1 basename|cut -d '_' -f 1|sort|uniq)
popd

for hash in $hashes
do
    /opt/anaconda3/bin/python -c "import csv_file; csv_file.download('$hash', '$target_dir')"
done