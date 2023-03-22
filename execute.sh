export LD_LIBRARY_PATH=/home/data3/czy3d/tmp/envs/pytorch3d/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
python main_train.py
while [[  $? -ne 0 ]];
do
	python main_train.py
done
