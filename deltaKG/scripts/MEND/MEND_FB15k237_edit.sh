for EDITS in 2
do
CUDA_VISIBLE_DEVICES=2 python models/MEND/run.py --n_edits ${EDITS} --config_file models/MEND/config_fb15k237_edit.yaml
done