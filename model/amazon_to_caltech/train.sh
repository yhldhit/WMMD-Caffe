#!/usr/bin/env sh

file = "data/amazon_to_caltech/traina2c.bak" 
if [-f $file]
then
    cp data/amazon_to_caltech/traina2c.bak data/amazon_to_caltech/traina2c.txt
else
    cp data/amazon_to_caltech/traina2c.txt data/amazon_to_caltech/traina2c.bak
fi

./build/tools/caffe train -solver=models/amazon_to_caltech/solver.prototxt -weights=models/bvlc_reference_caffenet.caffemodel
#-weights=models/icml/amazon_to_dslr/bvlc_reference_caffenet.caffemodel

