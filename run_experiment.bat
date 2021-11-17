set NUM_CPUS=4
set RAY_ROOT=../ray
set ITER_PER_EPOCH=100
set CUDA_VISIBLE_DEVICES=0

python experiment.py %1 %2 --epoch %3