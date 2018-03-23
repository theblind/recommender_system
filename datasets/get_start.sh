#!/usr/bin/env bash

hash wget 2>/dev/null || { echo >&2 "Wget required.  Aborting."; exit 1; }
hash unzip 2>/dev/null || { echo >&2 "unzip required.  Aborting."; exit 1; }

wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip -o "ml-20m.zip"
