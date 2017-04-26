# coding: utf-8
#fine tune target_lamda
import os
from google.protobuf import text_format
caffe_root = '../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
import default

from caffe.proto import caffe_pb2 as pb
#this part need to modify according to your path
pwd = os.getcwd()
model_path = default.model_path
solver_config_path = default.solver_config_path
net_config_path = default.net_config_path
Solver_TUNE = False
Net_TUNE = True
Nreapt = default.Nreapt
solver_default = default.solver

mmd_default = default.mmd

iter_of_epoch = default.iter_of_epoch
num_class_default = default.num_class
mmd_lock_default = default.mmd_lock
entropy_weight_default = default.entropy_weight
entropy_thresh_default = default.entropy_thresh

kernel_default = default.kernel

#do not change the rest
if(not os.path.exists(solver_config_path)):
    #define solver.prototxt
    print solver_config_path + " not exist"
else:
    ########## solver
    print "sovler define"
    s = pb.SolverParameter()

    #read and parse solver prototxt
    solver_str = str(open(solver_config_path,'rb').read())
    text_format.Parse(solver_str,s)

    #set parameter
    s.test_iter[0] = solver_default['test_iter']
    s.max_iter = solver_default['max_iter']


if(not os.path.exists(net_config_path)):
    print net_config_path  +" not exist"
else:
    ############## net
    net = pb.NetParameter()

    #read and parse net file
    #read_net_file = open(net_config_path,'rb')
    net_str = str(open(net_config_path,'rb').read())
    text_format.Parse(net_str,net);

#lr_list = [10**(-i*0.5) for i in range(5,11)]
tlambda_list = [0,0.004,0.02,0.04, 0.06,0.08,0.1,0.3]

for tlambda in tlambda_list:
    print tlambda

    #set default values for solver
    for key in solver_default:
        set_command = "s."+str(key.format())+'=' + "solver_default[\"" + str(key.format())+"\"]"
        if (key=="test_iter"):
            set_command = "s."+str(key.format())+'[0]=' + "solver_default[\"" + str(key.format())+"\"]"
        exec(set_command)

    #fine tune solver in this part
    if Solver_TUNE:
        s.base_lr= lr

    #write to solver file
    output = text_format.MessageToString(s)
    out_file = open(solver_config_path,'w')
    out_file.write(output)
    out_file.close()


    #set default values for net
    for layer in net.layer:
        name = layer.name
        if name in mmd_default:
            mmd_lambda = mmd_default[name]
            layer.mmd_param.mmd_lambda = mmd_lambda
            layer.mmd_param.num_of_kernel = kernel_default
            layer.mmd_param.entropy_thresh = entropy_thresh_default
            layer.mmd_param.iter_of_epoch = iter_of_epoch
            layer.mmd_param.num_class = num_class_default
            layer.mmd_param.mmd_lock = mmd_lock_default

        if name in entropy_weight_default:
            loss_weight = entropy_weight_default[name]
            layer.loss_weight = loss_weight

   #fine tune net in this part
    if Net_TUNE:
        #fine tune target_lambda
        tlambda_dict = {
                'loss':t_lambda,
                }

        for layer in net.layer:
            name  = layer.name
            if name in tlambda_dict:
                target_lambda = tlambda_dict[name]
                layer.loss_param.target_lambda = target_lambda

    #write to net file
    output = text_format.MessageToString(net)
    out_file = open(net_config_path,'w')
    out_file.write(output)
    out_file.close()

    #solve
    os.chdir(caffe_root)
    script_name = default.script
    command_ = 'bash ' + script_name
    for iparrel in range(Nreapt):
        os.system(command_)
    os.chdir(pwd)

#read_solver_file.close()
#read_net_file.close()
#caffe.set_device(0)
#caffe.set_mode_gpu()
#solver = None
#solver = caffe.get_solver(solver_config_path)
#print s.max_iter
#for it in range(s.max_iter):
#    solver.step(1)




