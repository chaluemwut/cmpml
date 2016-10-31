#!/bin/sh
python cmp.py bagging social &
python cmp.py boosted social &
python cmp.py randomforest social &
python cmp.py nb social &
python cmp.py knn social &
python cmp.py decsiontree social &
python cmp.py svm social &