#!/bin/bash

DATA_DIR=../data/cornell_movie_dialogs_corpus/

export LC_CTYPE=C
export LANG=C

cat ${DATA_DIR}movie_lines.txt | \
    sed -E "s/	/ /g; s/<.+>|\*//g; s/ \+\+\+\\$\+\+\+ /	/g; s/(['\"\.,;:?!\`-])/ \1 /g; s/ +/ /g" \
    > ${DATA_DIR}temp_lines.txt

cat ${DATA_DIR}temp_lines.txt | cut -d$'\t' -f1,5 | \
    sed -E "s/[0-9]+ /<number> /g; s/ [0-9]+/ <number>/g" > ${DATA_DIR}processed_lines.txt

cat ${DATA_DIR}movie_conversations.txt | sed -e "s/ +++$+++ /	/g; s/[',]//g" | \
    cut -d$'\t' -f4 > ${DATA_DIR}temp_conversations.txt

python3 process_cornell.py

rm ${DATA_DIR}temp_lines.txt ${DATA_DIR}temp_conversations.txt
