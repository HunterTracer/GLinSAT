# Partial Grpah Matching Experiment in GLinSAT

To run the code, first install all the necessary packages with

pip install qpth cvxpylayers cvxpy linsatnet numpy scipy pandas torch tensorboardX easydict pyyaml xlrd xlwt pynvml pygmtools ortools
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
apt-get install -y findutils libhdf5-serial-dev git wget libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz && tar zxvf cmake-3.19.1.tar.gz
cd cmake-3.19.1 && ./bootstrap && make && make install
apt-get install -y ninja-build

Download VOC2011 dataset [1] from http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html and make sure it looks like data/PascalVOC/TrainVal/VOCdevkit/VOC2011
Download keypoint annotation for VOC2011 [2] from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz and make sure it looks like data/PascalVOC/annotations
Following the code in [3], we download the open-source pretrained graph matching model for training. Make sure it looks like weights/pretrained_params_vgg16_ngmv2_afat-i_voc.pt
You can optionally download the pretrained model from this anonymous link https://figshare.com/s/384aaa5b4b738876ad81.

[1] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88: 303–338, 2010.
[2] Lubomir Bourdev and Jitendra Malik. Poselets: Body part detectors trained using 3d human pose annotations. In 2009 IEEE 12th international conference on computer vision, pages 1365–1372. IEEE, 2009.
[3] R. Wang, Y. Zhang, Z. Guo, T. Chen, X. Yang, and J. Yan. Linsatnet: the positive linear satisfiability neural networks. In International Conference on Machine Learning, pages 36605–36625. PMLR, 2023.

## Run the experiment

You can reproduce the main result of our experiment by running the following command.

python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way dense_block_diag_apdagd_kkt --project_temp 0.1 --project_dtype float32 > dense_block_diag_apdagd_kkt_128_float32_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way dense_block_diag_apdagd_kkt --project_temp 0.01 --project_dtype float32 > dense_block_diag_apdagd_kkt_128_float32_0.01_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_block_diag_apdagd_kkt --project_temp 0.1 --project_dtype float32 > sparse_block_diag_apdagd_kkt_128_float32_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_block_diag_apdagd_kkt --project_temp 0.01 --project_dtype float32 > sparse_block_diag_apdagd_kkt_128_float32_0.01_final.log 2>&1

python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way dense_block_diag_apdagd_direct --project_temp 0.1 --project_dtype float64 > dense_block_diag_apdagd_direct_128_float64_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way dense_block_diag_apdagd_direct --project_temp 0.01 --project_dtype float64 > dense_block_diag_apdagd_direct_128_float64_0.01_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_block_diag_apdagd_direct --project_temp 0.1 --project_dtype float64 > sparse_block_diag_apdagd_direct_128_float64_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_block_diag_apdagd_direct --project_temp 0.01 --project_dtype float64 > sparse_block_diag_apdagd_direct_128_float64_0.01_final.log 2>&1

python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way linsat --project_temp 0.1 --project_dtype float32 --project_max_iter 100 > linsat_128_float32_0.1_100_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way linsat --project_temp 0.01 --project_dtype float32 --project_max_iter 100 > linsat_128_float32_0.01_100_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way linsat --project_temp 0.1 --project_dtype float32 --project_max_iter 500 > linsat_128_float32_0.1_500_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way linsat --project_temp 0.01 --project_dtype float32 --project_max_iter 500 > linsat_128_float32_0.01_500_final.log 2>&1

python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_linsat --project_temp 0.1 --project_dtype float32 --project_max_iter 100 > sparse_linsat_128_float32_0.1_100_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_linsat --project_temp 0.01 --project_dtype float32 --project_max_iter 100 > sparse_linsat_128_float32_0.01_100_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_linsat --project_temp 0.1 --project_dtype float32 --project_max_iter 500 > sparse_linsat_128_float32_0.1_500_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way sparse_linsat --project_temp 0.01 --project_dtype float32 --project_max_iter 500 > sparse_linsat_128_float32_0.01_500_final.log 2>&1

python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way cvxpylayers --project_temp 0.1 --project_dtype float32 > cvxpylayers_128_float32_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way cvxpylayers --project_temp 0.01 --project_dtype float32 > cvxpylayers_128_float32_0.01_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way qpth --project_temp 0.1 --project_dtype float64 > qpth_128_float64_0.1_final.log 2>&1
python -u train_eval.py --cfg experiments/vgg16_ngmv2_linsat_voc-all.yaml --project_way qpth --project_temp 0.01 --project_dtype float64 > qpth_128_float64_0.01_final.log 2>&1

When using qpth, if you encounter errors like "RuntimeError: torch.linalg.lu_factor: (Batch element 0): U[66,66] is zero and using it on lu_solve would result in a division by zero. If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or linalg.lu_factor_ex(A, pivot)", please try to change "qpth/solvers/pdipm/batch.py", line 9, in lu_hack
    data, pivots = torch.linalg.lu_factor(x, pivot=not x.is_cuda)
--->data, pivots = torch.linalg.lu_factor(x, pivot=True)
and then delete line 11 to 19.
