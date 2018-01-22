if [ -d "mpi_learn" ]; then
    git clone git@github.com:duanders/mpi_learn.git
fi
cd mpi_learn
git pull
cd ..
touch mpi_learn/__init__.py
