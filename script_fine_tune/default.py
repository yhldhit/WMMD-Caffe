model_path = "../models/webcam_to_caltech/"
solver_config_path = model_path + "solver.prototxt"
net_config_path = model_path + "train_val.prototxt"
Nreapt = 2

#fistly you need to set the solver and net parameter for default

solver = {
    'test_iter':14,
    'test_interval': 200,
    'base_lr':0.0003162,
    'max_iter':2200,
    'momentum':0.9,
    'stepsize':500,
    'snapshot':2200,
    'snapshot_prefix':model_path[1:]+"mmd",
    'net':net_config_path[1:],
}


mmd = {
    #work in alex net
    "mmd_fc7":0.0,
    "mmd_fc8":0.0,
    #work in googlenet
    "mmd_loss3":0.3,
    "mmd_4d":2,
    "mmd_4e":3,
}
iter_of_epoch = 22
num_class = 10
mmd_lock = 1
entropy_weight = {
    "entropy_loss":0,
    }
entropy_thresh = 10.0

kernel = 10

script = model_path[3:] + 'train.sh'

