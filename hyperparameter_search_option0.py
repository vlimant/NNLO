from skopt import Optimizer
import subprocess
import time
import mpiLAPI
import numpy as np
import json
import os

def dummy_model(params):
    import random
    time.sleep(1)
    return random.randint(0, 10)


def train_model_dummy(params):
    n_nodes = "1"
    model_arch_files = "mnist_arch.json"
    train_files = "train_mnist.list"
    test_files = "test_mnist.list"
    n_epochs = "1"

    # model = build_model(params)
    # json_model = model.to_json()
    json_model = "{}"

    mpi_cmd = "mpirun"  # "mpirun"
    train_cmd = mpi_cmd + " -np " + n_nodes + " python ./mpi_learn/MPIDriver.py " + model_arch_files + " " + train_files + " " + \
        test_files + " --loss categorical_crossentropy --tf --epochs " + n_epochs

    try:
        print("starting MPI process")
        print(train_cmd)
        time.sleep(1)  # don't overload the cluster with mpi_run calls
        output = subprocess.check_output(train_cmd, shell=True)
        search_str = "Wrote trial information"
        for line in output.split('\n'):
            print("search output: " + line)

            if line.startswith(search_str):
                print("found line: " + line)
                fname = line.split(' ')[-1]
                with open(fname) as f:
                    hist = json.load(f)
                    fom = hist["history"]["0"]["val_loss"][-1]
                    print("fom: " + str(fom))
                    return float(fom)
    except subprocess.CalledProcessError as err:
        print("CalledProcessError: {}".format(err))


def train_model(params, n_nodes=1, n_epochs=3):
    #model = mpiLAPI.test_densenet(depth=3*params[0]+4, growth_rate=params[1], dropout_rate=params[2], nb_filter=params[3], lr=10.0**np.int32(params[4]))
    model = mpiLAPI.test_cnn(dropout=params[0], kernel_size=params[1], lr=10.**np.int32(params[2]))   
    mlapi = mpiLAPI.mpi_learn_api( model = model,
                           train_pattern = '/bigdata/shared/LCDJets_Remake/train/04*.h5',
                           val_pattern = '/bigdata/shared/LCDJets_Remake/val/020*.h5',
                           check_file = True
                           )
    try:
        output = mlapi.train(N=n_nodes,
                    trial_name = 'test',
                    features_name = 'Images',
                    labels_name = 'Labels',
                    batch = 2,
                    epoch = n_epochs,
                    verbose = True,
                    loss = 'categorical_crossentropy',
                    easgd = False,
                    early_stopping = 5
                    )
        search_str = "Wrote trial information"
        for line in output.split('\n'):
            print("search output: " + line)

            if line.startswith(search_str):
                print("found line: " + line)
                fname = line.split(' ')[-1]
                with open(fname) as f:
                    hist = json.load(f)
                    fom = hist["history"]["0"]["val_loss"][-1]
                    print("fom: " + str(fom))
                    return float(fom)
    except subprocess.CalledProcessError as err:
        print("CalledProcessError: {}".format(err))
        return np.inf


if __name__ == '__main__':
    n_trials = 4
    os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

    concurrent_train = 5
    param_grid_cnn = [
        (.0, .85),  # dropout_rate
        (3,10),  # kernel_size
        (-5, 1),  # lr
    ]
    param_grid_densenet = [
        (3, 8),  # depth
        (1, 5),  # growth_rate
        (.0, .85),  # dropout_rate
        (32, 256),  # nb_filters
        (-5, 1),  # lr
    ]

    # https://scikit-optimize.github.io/#skopt.Optimizer
    bayesian_opt = Optimizer(param_grid_cnn)

    # TODO: how many do you want to spawn at the start?
    for i in range(n_trials):
        suggested = bayesian_opt.ask()
        print(suggested)
        y = train_model(suggested)
        print(y)
        # y = dummy_model(suggested)
        bayesian_opt.tell(suggested, y)
        print('iteration:', i, suggested, y)
