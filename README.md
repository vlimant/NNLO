# mpi_opt
Bayesian optimization on supercomputer with MPI


# alpha instructions

Install the software together with mpi_learn
```
git clone git@github.com:thongonary/mpi_opt.git
git clone git@github.com:svalleco/mpi_learn.git
cd mpi_opt
ln -s ../mpi_learn/mpi_learn
```

To run the mnist example (you need to get the mnist data file first using get_mnist in mpi_learn) with 4 blocks of 5 processes for 10 epoches, and 10 bayesian optimization cycle : (1 opt master + 4 block x (1 master + 4 workers)) = 21 processes

```
mpirun -tag-output -n 21  python3 hyperparameter_search_option5.py --block-size 5 --example mnist --epochs 10 --num-iterations 10
```

To run with 5-fold cross validation : (1 opt master + (5 fold x (4 block x (1 master + 4 workers))) = 101
```
mpirun -tag-output -n 101 python3 hyperparameter_search_option5.py --block-size 5 --example mnist --epochs 10 --num-iterations 10 --n-fold 5
```

To run with a genetic algorithm instead of Bayesian optimization (note that the 
```
mpirun -tag-output -n 21  python3 hyperparameter_search_option5.py --block-size 5 --example mnist --epochs 10 --num-iterations 10 --ga
```