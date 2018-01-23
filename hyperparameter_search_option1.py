from skopt import Optimizer
import subprocess
import time
import mpiLAPI
import numpy as np
import json


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


def get_fom_from_json_file(fname):
    with open(fname) as f:
        hist = json.load(f)
        fom = hist["history"]["0"]["val_loss"][-1]
        print("fom: " + str(fom))
        return float(fom)


def train_model(params, n_nodes=3, n_epochs=3):
    model = mpiLAPI.get_model(depth=3*params[0]+4, growth_rate=params[1], dropout_rate=params[2], nb_filter=params[3], lr=10.0**np.int32(params[4]))
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
                    batch = 4,
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
                return get_fom_from_json_file(fname)
    except subprocess.CalledProcessError as err:
        print("CalledProcessError: {}".format(err))
        return np.inf


def get_train_cmd(self, **args):
    com = 'mpirun -n %d mpi_learn/MPIDriver.py %s %s %s'%(
        args.get('N', 2),
        self.json_file,
        self.train_files,
        self.val_files
    )
    for option,default in { 'trial_name' : 'mpi_run',
                                'master_gpu' : True,
                                'features_name' : 'X',
                                'labels_name' : 'Y',
                                'epoch' : 100,
                                'batch' : 100,
                                'loss' : 'categorical_crossentropy',
                                'verbose': False,
                                'early_stopping' : False,
                                'easgd' : False,
                                'tf': True,
                                'elastic_force': 0.9,
                                'elastic_momentum': 0.99,
                                'elastic_lr':0.001,
                                }.items():
        v = args.get(option,default)
        if type(v)==bool:
            com +=' --%s'%option.replace('_','-') if v else ''
        else:
            com+=' --%s %s'%(option.replace('_','-'), v)
    print(com)
    return com


def train_model_async(params, n_nodes=3, n_epochs=3, trial_id=0):
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
        test_files + " --loss categorical_crossentropy --tf --epochs " + n_epochs + " --trial-name " + str(trial_id)

    print("starting MPI process asynchronously.")
    print(train_cmd)
    time.sleep(1)  # don't overload the cluster with mpi_run calls

    import tempfile
    tfil = tempfile.TemporaryFile()

    p = subprocess.Popen(train_cmd, shell=True, stdout=tfil, stderr=tfil)
    return p


def get_finished_process(proc_list):
    print("waiting for process to stop training")
    while True:
        for run_id, params, proc in proc_list:
            if proc.poll() == 0:
                print("process finished: " + str(run_id))
                return (run_id, params, proc)
        time.sleep(10)


if __name__ == '__main__':
    n_trials = 10
    concurrent_train = 5
    model_name = "mnist_arch"
    param_grid = [
        (3, 8),  # depth
        (1, 5),  # growth_rate
        (.0, .5),  # dropout_rate
        (32, 256),  # nb_filters
        (-5, 1),  # lr
    ]

    # https://scikit-optimize.github.io/#skopt.Optimizer
    bayesian_opt = Optimizer(param_grid)

    # TODO: how many do you want to spawn at the start?
    running_proc = []
    run_id = 0
    while run_id < n_trials:
        print("new run: " + str(run_id))
        while len(running_proc) < concurrent_train:
            print("new run: " + str(run_id))
            suggested = bayesian_opt.ask()
            print(suggested)
            p_i = train_model_async(suggested, n_epochs=1, n_nodes=1, trial_id=run_id)
            print((run_id, suggested, p_i))
            running_proc.append((run_id, suggested, p_i))
            run_id += 1

        # infinite loop until some configuration to finish
        run_completed, par_i, proc_id = get_finished_process(running_proc)
        running_proc.remove((run_completed, par_i, proc_id))  # remove completed process
        fname = '_'.join([model_name, str(run_completed), "history.json"])
        y = get_fom_from_json_file(fname)
        result = bayesian_opt.tell(suggested, y)
        print('completed iteration:', run_completed, par_i, y)

    print("Best parameters: " + str(result.x))
