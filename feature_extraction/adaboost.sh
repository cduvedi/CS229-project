#!/bin/tcsh

setenv PYTHONPATH ${HOME}:${HOME}/lib/python2.7/site-packages/
setenv FULLPATH /afs/.ir.stanford.edu/users/b/h/bhrugu/CS229/project/temp/CS229-project/feature_extraction
python ${FULLPATH}/adaboost.py
