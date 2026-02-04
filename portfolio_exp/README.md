# Portfolio Allocation Experiment in GLinSAT

To run the code, first install all the necessary packages with

pip install qpth cvxpylayers cvxpy linsatnet numpy scipy pandas torch

## Run the experiment

You can reproduce the main result of our portfolio optimization experiment by running the following command

python -u run_train.py --project_way dense_apdagd_direct --temp 0.1 --project_dtype float32 > dense_apdagd_direct_float32_0.1.log 2>&1
python -u run_train.py --project_way dense_apdagd_direct --temp 0.01 --project_dtype float32 > dense_apdagd_direct_float32_0.01.log 2>&1
python -u run_train.py --project_way sparse_apdagd_direct --temp 0.1 --project_dtype float32 > sparse_apdagd_direct_float32_0.1.log 2>&1
python -u run_train.py --project_way sparse_apdagd_direct --temp 0.01 --project_dtype float32 > sparse_apdagd_direct_float32_0.01.log 2>&1

python -u run_train.py --project_way dense_apdagd_kkt --temp 0.1 --project_dtype float32 > dense_apdagd_kkt_float32_0.1.log 2>&1
python -u run_train.py --project_way dense_apdagd_kkt --temp 0.01 --project_dtype float32 > dense_apdagd_kkt_float32_0.01.log 2>&1
python -u run_train.py --project_way sparse_apdagd_kkt --temp 0.1 --project_dtype float32 > sparse_apdagd_kkt_float32_0.1.log 2>&1
python -u run_train.py --project_way sparse_apdagd_kkt --temp 0.01 --project_dtype float32 > sparse_apdagd_kkt_float32_0.01.log 2>&1

python -u run_train.py --project_way linsat --temp 0.1 --max_iter 100 --project_dtype float32 > linsat_100_float32_0.1.log 2>&1
python -u run_train.py --project_way linsat --temp 0.1 --max_iter 500 --project_dtype float32 > linsat_500_float32_0.1.log 2>&1
python -u run_train.py --project_way linsat --temp 0.01 --max_iter 100 --project_dtype float32 > linsat_100_float32_0.01.log 2>&1
python -u run_train.py --project_way linsat --temp 0.01 --max_iter 500 --project_dtype float32 > linsat_500_float32_0.01.log 2>&1

python -u run_train.py --project_way sparse_linsat --temp 0.1 --max_iter 100 --project_dtype float32 > sparse_linsat_100_float32_0.1.log 2>&1
python -u run_train.py --project_way sparse_linsat --temp 0.1 --max_iter 500 --project_dtype float32 > sparse_linsat_500_float32_0.1.log 2>&1
python -u run_train.py --project_way sparse_linsat --temp 0.01 --max_iter 100 --project_dtype float32 > sparse_linsat_100_float32_0.01.log 2>&1
python -u run_train.py --project_way sparse_linsat --temp 0.01 --max_iter 500 --project_dtype float32 > sparse_linsat_500_float32_0.01.log 2>&1

python -u run_train.py --project_way qpth --temp 0.1 --project_dtype float32 > qpth_float32_0.1.log 2>&1
python -u run_train.py --project_way qpth --temp 0.01 --project_dtype float32 > qpth_float32_0.01.log 2>&1
python -u run_train.py --project_way cvxpylayers --temp 0.1 --project_dtype float32 > cvxpylayers_float32_0.1.log 2>&1
python -u run_train.py --project_way cvxpylayers --temp 0.01 --project_dtype float32 > cvxpylayers_float32_0.01.log 2>&1

