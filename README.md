# WMMD-Caffe
This is an implementation of CVPR17 paper "Mind the Class Weight Bias: Weighted Maximum Mean Discrepancy for Unsupervised Domain Adaptation". We fork the repository from [mmd-caffe](https://github.com/zhuhan1236/mmd-caffe) and make our modifications. We show them below:
- mmd layer: the backward function in mmd_layer.cu is adjusted to replace the conventional mmd with weighted mmd specified in the [paper](wmmd)
- softmax loss layer: Instead of ignoring the empirical loss on target domain, we modified the function so that logistic loss is added based on pseudo label. 
- data layer: we add an parameter to the data label so that the number of classes is conveyed to mmd layer. 

# 
The machines configuration that run the experiments are specificied below:
- OS: UNIX
- GPU: TiTan X
- CUDA: version 7.5
- CUDNN: version 3.0

Slight changes may not results instabilities. 

# Usage
* prepare model: The `bvlc_reference_caffenet` and `bvlc_googlenet` are used as initialzation for Alexnet and GoogleNet, respectively. They can be download from [here](http://caffe.berkeleyvision.org/model_zoo.html). The model structure is specified in the relative path `model/task_name/train_val.prototxt`, *e.g.,* when we transfer from amazon to caltech in *office-10+caltech-10* dataset, replace the task_name with `amazon_to_caltech`. 
* prepare data: Since we reach the raw images on the disk when training, all the images file path need to be written into the `.txt` file, kept in `data/task_name/` directory. For example, data/amazon_to_caltech/train\(value\).txt. To constrcuct such txtfile, a python script is offered in file `images/code/data_constructor.py`. Following are the main steps:
  * change into the directory `image/code/`: `cd image/code`
  * `python data_constructor.py`. You will be asked to input the source and target domain name, e.g., `amazon` and `caltech`, respectively.
  * the generated file traina2c.txt and texta2c.txt could be found in the parent directory, copy them into `../data/amazon2caltech/`: `cp *a2c.txt ../amazon2caltech`.
*  fine tune a model: To fine tune the model paramenters, run the shell `./model/amazon2caltech/train.sh`
You will see the accuracy results once the script is finished. The detailed results could be found in files that are stored in the `./log`. And the tuned model will be stored in `model/amazon2caltech/`

# Fine tune the model parameters 
Three model parameters are tunned in our experiments, _i.e.,_ $\lambda$, $\beta$, $lr$ and please refer to the paper for the meaning of these parameters. Manually tuning them in net defination file, _i.e.,_ 'trainval.protxt', could be rather tedious. We provide python scripts to help tune these parameters which can be found in directory `script_fine_tune/`. To correctly run these scripts, please follow the instruction to install the Python runtime library of [Protobuf](https://github.com/google/protobuf). Also the Caffe python's interface should be compiled: `make pycaffe`.

# Citation
```
```

