#!/usr/bin/env sh

file = "data/amazon_to_caltech/traina2c.bak" 
if [-f $file]
then
    cp data/amazon_to_caltech/traina2c.bak data/amazon_to_caltech/traina2c.txt
else
    cp data/amazon_to_caltech/traina2c.txt data/amazon_to_caltech/traina2c.bak
fi

./build/tools/caffe train -solver=models/amazon_to_caltech/googlenet_solver.prototxt -weights=models/bvlc_googlenet.caffemodel
cp data/amazon_to_caltech/traina2c.bak data/amazon_to_caltech/traina2c.txt
