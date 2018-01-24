from skopt import Optimizer
import subprocess
import time
import mpiLAPI
import numpy as np
import json
import base_model


def get_fom_from_json_file(fname):
    with open(fname) as f:
        hist = json.load(f)
        fom = hist["history"]["0"]["val_loss"][-1]
        print("fom: " + str(fom))
        return float(fom)


def get_train_cmd(mlapi, **args):
    com = 'mpirun -n %d python3 mpi_learn/MPIDriver.py %s %s %s' % (
        args.get('N', 2),
        mlapi.json_file,
        mlapi.train_files,
        mlapi.val_files
    )
    for option, default in {'trial_name': 'mpi_run',
                            'master_gpu': True,
                            'features_name': 'X',
                            'labels_name': 'Y',
                            'epoch': 100,
                            'batch': 100,
                            'loss': 'categorical_crossentropy',
                            'verbose': False,
                            'early_stopping': False,
                            'easgd': False,
                            'tf': True,
                            'elastic_force': 0.9,
                            'elastic_momentum': 0.99,
                            'elastic_lr': 0.001,
                            }.items():
        v = args.get(option, default)
        if type(v) == bool:
            com += ' --%s' % option.replace('_', '-') if v else ''
        else:
            com += ' --%s %s' % (option.replace('_', '-'), v)
    return com


def train_model_async(json_model, n_nodes=3, n_epochs=3, trial_id=0, model_name='cnn_arch'):
    print("Starting MPI process asynchronously.")
    mlapi = mpiLAPI.mpi_learn_api(model=json_model,
                                  train_pattern='/bigdata/shared/LCDJets_Remake/train/04*.h5',
                                  val_pattern='/bigdata/shared/LCDJets_Remake/val/020*.h5',
                                  check_file=True,
                                  nohash=True,
                                  model_name=model_name
                                  )
    train_cmd = get_train_cmd(mlapi, N=n_nodes,
                              trial_name=str(trial_id),
                              features_name='Images',
                              labels_name='Labels',
                              batch=4,
                              epoch=n_epochs,
                              verbose=True,
                              loss='categorical_crossentropy',
                              easgd=False,
                              early_stopping=5
                              )

    print(train_cmd)
    time.sleep(5)  # don't overload the cluster with mpi_run calls

    if not verbose:
        import tempfile  # used only to suppress stdout and stderr output
        tfil = tempfile.TemporaryFile()
        p = subprocess.Popen(train_cmd, shell=True, stdout=tfil, stderr=tfil)
    else:
        p = subprocess.Popen(train_cmd, shell=True)
    return p


def get_finished_process(proc_list):
    print("waiting for process to stop training")
    while True:
        for run_id, params, proc in proc_list:
            if proc.poll() == 0:
                print("process finished: " + str(run_id))
                running_proc.remove((run_id, params, proc))
                return (run_id, params, proc)
            elif proc.poll() != 0 and proc.poll() is not None:
                print("process finished with errors: " + str(run_id))
                running_proc.remove((run_id, params, proc))
        time.sleep(10)


if __name__ == '__main__':
    n_trials = 1
    concurrent_train = 5
    model = base_model.CNNModel()
    param_grid = model.get_parameter_grid()
    bayesian_opt = Optimizer(param_grid)

    running_proc = []
    run_id = 0
    while run_id < n_trials:
        print("new run: " + str(run_id))
        while len(running_proc) < concurrent_train:
            print("new run: " + str(run_id))
            suggested = bayesian_opt.ask()
            print(suggested)
            p_i = train_model_async(model.build(suggested), n_epochs=3, n_nodes=2, trial_id=run_id,
                                    model_name=model.get_name(), verbose=True)
            print((run_id, suggested, p_i))
            running_proc.append((run_id, suggested, p_i))
            run_id += 1

        # infinite loop until some configuration to finish
        run_completed, par_i, proc_id = get_finished_process(running_proc)
        running_proc.remove((run_completed, par_i, proc_id))  # remove completed process
        fname = '_'.join([model.get_name(), str(run_completed), "history.json"])
        y = get_fom_from_json_file(fname)
        result = bayesian_opt.tell(suggested, y)
        print('completed iteration:', run_completed, par_i, y)

    print("Best parameters: " + str(result.x))
