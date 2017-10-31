#!/bin/sh
notebook=$1
script=$2
jupyter nbconvert --to script --execute "${notebook}" && python "${script}"
rm "${script}"