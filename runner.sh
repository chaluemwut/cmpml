#!/bin/sh
python cmp.py bagging &
python cmp.py boosted &
python cmp.py randomforest &
python cmp.py nb &
python cmp.py decsiontree &
python cmp.py knn &
python cmp.py svm &