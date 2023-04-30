for EDITS in 2
do
CUDA_VISIBLE_DEVICES=1 python models/MEND/run.py --n_edits ${EDITS} --config_file models/MEND/config_wn18rr_add.yaml
done