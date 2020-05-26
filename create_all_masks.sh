#!/bin/bash

# create training maskes (features) from all files in IFP Samples


target_dir=$1
pushd "$target_dir"
hashes=$(ls *.xml|cut -d '_' -f 1|sort|uniq)
popd

for hash in $hashes
do
    pushd "$target_dir"
    pages=$(ls ${hash}_*.csv|cut -d '_' -f 2|sort -n)
    popd

    echo $hash
    echo $pages

    for page in $pages
    do
        python split_lines.py $hash $page
    done


done