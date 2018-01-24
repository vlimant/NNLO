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


def train_model_async(json_model, n_nodes=3, n_epochs=3, trial_id=0, get_output=True, model_name='cnn_arch'):
    print("Starting MPI process asynchronously.")
    mlapi = mpiLAPI.mpi_learn_api(model=json_model,
                                  train_pattern='/bigdata/shared/LCDJets_Remake/train/04*.h5',
                                  val_pattern='/bigdata/shared/LCDJets_Remake/val/020*.h5',
                                  check_file=True,
                                  nohash=True,
                                  model_name=model_name
                                  )
    p = mlapi.train_async(get_output=get_output, N=n_nodes,
                trial_name = str(trial_id),
                features_name = 'Images',
                labels_name = 'Labels',
                batch = 4,
                epoch = n_epochs,
                verbose = True,
                loss = 'categorical_crossentropy',
                easgd = False,
                early_stopping = 5
    )
    time.sleep(5)  # don't overload the cluster with mpi_run calls
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
    os.environ["CUDA_VISIBLE_DEVICES"]="0,3,4,5"
    n_trials = 5 
    concurrent_train = 3
    model = base_model.FunctionalModel()
    param_grid = model.get_parameter_grid()
    bayesian_opt = Optimizer(param_grid)

    running_proc = []
    run_id = 0
    while run_id < n_trials:
        #print("new run: " + str(run_id))
        while len(running_proc) < concurrent_train:
            print("new run: " + str(run_id))
            suggested = bayesian_opt.ask()
            print(suggested)
            p_i = train_model_async(model.build(suggested), n_epochs=3, n_nodes=2, trial_id=run_id,
                                    get_output=True, model_name=model.get_name())
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
