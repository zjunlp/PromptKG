for EDITS in 2
do
CUDA_VISIBLE_DEVICES=0 python models/MEND/run.py --n_edits ${EDITS} --config_file models/MEND/config_fb15k237_add.yaml
done