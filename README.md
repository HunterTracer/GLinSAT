# TSP Experiment in GLinSAT

To run the code, first install all the necessary packages with

pip install qpth cvxpylayers cvxpy linsatnet numpy scipy pandas torch

## Run the experiment

You can reproduce the main result of our experiment by running the following command.

Command for training: (note that qpth and cvxpylayers will be significantly time-consuming)

python -u run_train.py --task StartEnd --project_way dense_apdagd_direct --temp 0.1 > StartEnd_dense_apdagd_direct_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way dense_apdagd_direct --temp 0.01 > StartEnd_dense_apdagd_direct_0.01.log 2>&1
python -u run_train.py --task StartEnd --project_way dense_apdagd_kkt --temp 0.1 > StartEnd_dense_apdagd_kkt_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way dense_apdagd_kkt --temp 0.01 > StartEnd_dense_apdagd_kkt_0.01.log 2>&1
python -u run_train.py --task Priority --project_way dense_apdagd_direct --temp 0.1 > Priority_dense_apdagd_direct_0.1.log 2>&1
python -u run_train.py --task Priority --project_way dense_apdagd_direct --temp 0.01 > Priority_dense_apdagd_direct_0.01.log 2>&1
python -u run_train.py --task Priority --project_way dense_apdagd_kkt --temp 0.1 > Priority_dense_apdagd_kkt_0.1.log 2>&1
python -u run_train.py --task Priority --project_way dense_apdagd_kkt --temp 0.01 > Priority_dense_apdagd_kkt_0.01.log 2>&1

python -u run_train.py --task StartEnd --project_way sparse_apdagd_direct --temp 0.1 > StartEnd_sparse_apdagd_direct_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_apdagd_direct --temp 0.01 > StartEnd_sparse_apdagd_direct_0.01.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_apdagd_kkt --temp 0.1 > StartEnd_sparse_apdagd_kkt_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_apdagd_kkt --temp 0.01 > StartEnd_sparse_apdagd_kkt_0.01.log 2>&1
python -u run_train.py --task Priority --project_way sparse_apdagd_direct --temp 0.1 > Priority_sparse_apdagd_direct_0.1.log 2>&1
python -u run_train.py --task Priority --project_way sparse_apdagd_direct --temp 0.01 > Priority_sparse_apdagd_direct_0.01.log 2>&1
python -u run_train.py --task Priority --project_way sparse_apdagd_kkt --temp 0.1 > Priority_sparse_apdagd_kkt_0.1.log 2>&1
python -u run_train.py --task Priority --project_way sparse_apdagd_kkt --temp 0.01 > Priority_sparse_apdagd_kkt_0.01.log 2>&1

python -u run_train.py --task StartEnd --project_way linsat --temp 0.1 --max_iter 100 > StartEnd_linsat_0.1_100.log 2>&1
python -u run_train.py --task StartEnd --project_way linsat --temp 0.01 --max_iter 100 > StartEnd_linsat_0.01_100.log 2>&1
python -u run_train.py --task StartEnd --project_way linsat --temp 0.1 --max_iter 500 > StartEnd_linsat_0.1_500.log 2>&1
python -u run_train.py --task StartEnd --project_way linsat --temp 0.01 --max_iter 500 > StartEnd_linsat_0.01_500.log 2>&1
python -u run_train.py --task Priority --project_way linsat --temp 0.1 --max_iter 100 > Priority_linsat_0.1_100.log 2>&1
python -u run_train.py --task Priority --project_way linsat --temp 0.01 --max_iter 100 > Priority_linsat_0.01_100.log 2>&1
python -u run_train.py --task Priority --project_way linsat --temp 0.1 --max_iter 500 > Priority_linsat_0.1_500.log 2>&1
python -u run_train.py --task Priority --project_way linsat --temp 0.01 --max_iter 500 > Priority_linsat_0.01_500.log 2>&1

python -u run_train.py --task StartEnd --project_way sparse_linsat --temp 0.1 --max_iter 100 > StartEnd_sparse_linsat_0.1_100.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_linsat --temp 0.01 --max_iter 100 > StartEnd_sparse_linsat_0.01_100.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_linsat --temp 0.1 --max_iter 500 > StartEnd_sparse_linsat_0.1_500.log 2>&1
python -u run_train.py --task StartEnd --project_way sparse_linsat --temp 0.01 --max_iter 500 > StartEnd_sparse_linsat_0.01_500.log 2>&1
python -u run_train.py --task Priority --project_way sparse_linsat --temp 0.1 --max_iter 100 > Priority_sparse_linsat_0.1_100.log 2>&1
python -u run_train.py --task Priority --project_way sparse_linsat --temp 0.01 --max_iter 100 > Priority_sparse_linsat_0.01_100.log 2>&1
python -u run_train.py --task Priority --project_way sparse_linsat --temp 0.1 --max_iter 500 > Priority_sparse_linsat_0.1_500.log 2>&1
python -u run_train.py --task Priority --project_way sparse_linsat --temp 0.01 --max_iter 500 > Priority_sparse_linsat_0.01_500.log 2>&1

python -u run_train.py --task StartEnd --project_way cvxpylayers --temp 0.1 > StartEnd_cvxpylayers_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way cvxpylayers --temp 0.01 > StartEnd_cvxpylayers_0.01.log 2>&1
python -u run_train.py --task Priority --project_way cvxpylayers --temp 0.1 > Priority_cvxpylayers_0.1.log 2>&1
python -u run_train.py --task Priority --project_way cvxpylayers --temp 0.01 > Priority_cvxpylayers_0.01_100.log 2>&1

python -u run_train.py --task StartEnd --project_way qpth --temp 0.1 > StartEnd_qpth_0.1.log 2>&1
python -u run_train.py --task StartEnd --project_way qpth --temp 0.01 > StartEnd_qpth_0.01.log 2>&1
python -u run_train.py --task Priority --project_way qpth --temp 0.1 > Priority_qpth_0.1.log 2>&1
python -u run_train.py --task Priority --project_way qpth --temp 0.01 > Priority_qpth_0.01_100.log 2>&1

Example Command for evaluate:

python -u run_evaluate.py --model_dict outputs/dense_apdagd_direct_StartEnd_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_StartEnd_dense_apdagd_direct_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_direct_StartEnd_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_StartEnd_dense_apdagd_direct_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_direct_Priority_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_Priority_dense_apdagd_direct_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_direct_Priority_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_Priority_dense_apdagd_direct_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/dense_apdagd_kkt_StartEnd_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_StartEnd_dense_apdagd_kkt_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_kkt_StartEnd_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_StartEnd_dense_apdagd_kkt_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_kkt_Priority_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_Priority_dense_apdagd_kkt_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/dense_apdagd_kkt_Priority_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_Priority_dense_apdagd_kkt_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/sparse_apdagd_direct_StartEnd_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_StartEnd_sparse_apdagd_direct_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_direct_StartEnd_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_StartEnd_sparse_apdagd_direct_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_direct_Priority_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_Priority_sparse_apdagd_direct_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_direct_Priority_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_Priority_sparse_apdagd_direct_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/sparse_apdagd_kkt_StartEnd_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_StartEnd_sparse_apdagd_kkt_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_kkt_StartEnd_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_StartEnd_sparse_apdagd_kkt_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_kkt_Priority_20_0.1  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_Priority_sparse_apdagd_kkt_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_apdagd_kkt_Priority_20_0.1  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_Priority_sparse_apdagd_kkt_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/linsat_StartEnd_20_0.1_100  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_100_StartEnd_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_StartEnd_20_0.1_100  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_100_StartEnd_linsat_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_Priority_20_0.1_100  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_100_Priority_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_Priority_20_0.1_100  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_100_Priority_linsat_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/linsat_StartEnd_20_0.1_500  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_500_StartEnd_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_StartEnd_20_0.1_500  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_500_StartEnd_linsat_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_Priority_20_0.1_500  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_500_Priority_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/linsat_Priority_20_0.1_500  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_500_Priority_linsat_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/sparse_linsat_StartEnd_20_0.1_100  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_100_StartEnd_sparse_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_linsat_StartEnd_20_0.1_100  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_100_StartEnd_sparse_linsat_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_linsat_Priority_20_0.1_100  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_100_Priority_sparse_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_linsat_Priority_20_0.1_100  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_100_Priority_sparse_linsat_0.1_beam_search.log 2>&1

python -u run_evaluate.py --model_dict outputs/sparse_linsat_StartEnd_20_0.1_500  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_500_StartEnd_sparse_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs/sparse_linsat_StartEnd_20_0.1_500  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_500_StartEnd_sparse_linsat_0.1_beam_search.log 2>&1
python -u run_evaluate.py --model_dict outputs_eval/sparse_linsat_Priority_20_0.1_500  --test_epoch 50 --postprocess_way rounding --temp 0.01 > eval_0.1_500_Priority_sparse_linsat_0.01_rounding.log 2>&1
python -u run_evaluate.py --model_dict outputs_eval/sparse_linsat_Priority_20_0.1_500  --test_epoch 50 --postprocess_way beam_search --temp 0.1 > eval_0.1_500_Priority_sparse_linsat_0.1_beam_search.log 2>&1

