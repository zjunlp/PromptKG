#SBATCH -N 1
#SBATCH -n 5
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue


source activate xx


CUDA_VISIBLE_DEVICES=0 python main.py --gpus "0," --max_epochs=50  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --accumulate_grad_batches 1 \
   --model_class KGBERTUseLabelWord \
   --batch_size 16 \
   --check_val_every_n_epoch 3 \
   --data_dir dataset/WN18RR/666 \
   --max_seq_length 70 \
   --lr 5e-5 \
   --eval_batch_size 2500 \
   --chunk 1