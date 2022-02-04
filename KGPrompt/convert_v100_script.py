# TODO turn the ordinary script to slrum script
import argparse
import os

parser = argparse.ArgumentParser(description='script path')

parser.add_argument('-script_path',  default="./scripts/run.sh", type=str,help='an integer for the accumulator')
parser.add_argument('-gpus', type=int, default=1, help='an integer for the accumulator')
parser.add_argument('-cpus', type=int, default=5, help='an integer for the accumulator')
parser.add_argument('-memory', type=int, default=16, choices=[16,32], help='an integer for the accumulator')
parser.add_argument('-envs', type=str, default="xx", help='an integer for the accumulator')
args = parser.parse_args()


# use16 means use the swarm
use16 = args.memory == 16

base = f"""
#SBATCH -N {args.gpus}
#SBATCH -n {args.cpus}
#SBATCH -M {"swarm" if use16 else "priv"}
#SBATCH -p {"gpu" if use16 else "priv_para"}
#SBATCH --gres=gpu:1
#SBATCH --no-requeue


source activate {args.envs}
"""




lines = []
with open(args.script_path, "r") as file:
	# read the file lines
	lines = file.readlines()
	

if "#" not in lines[0]:
	lines.insert(0, "#!/bin/bash")
# add base code and envs code 
lines.insert(1, base)
for idx, line in enumerate(lines):
	if "CUDA_VISIBLE_DEVICES=1" in line:
		lines[idx] = line.replace("CUDA_VISIBLE_DEVICES=1", "")
	if "model_name_or_path" in line:
		lines[idx] = line.replace("bert-base-uncased", "/home/zhangningyu/project/xx/pretrained_model/bert-base-uncased ")

lines[-1] = lines[-1] + "\\\n"
for i in range(10):
	# split into 10 chunks to speed up testing
	t_lines = lines + [f"	--chunk {i}"]
	output_file_path = args.script_path.replace(".sh", "_srun.sh")
	with open(output_file_path, "w") as file:
		file.writelines(t_lines)

	os.system(f"sbatch {output_file_path}")
	print(t_lines)



