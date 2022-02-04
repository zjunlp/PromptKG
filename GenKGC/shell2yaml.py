import os
import sys
import yaml

path = sys.argv[1]
output_path = sys.argv[2]
assert os.path.exists(path)
with open(path) as file:
    text = file.read()

my_args = text.split("#")[0].split("--")[1:]
print(my_args)
total_args = {}
for args in my_args:
    if "gpu" in args: continue # ignore the CUDA setting
    if "overwrite_cache" in args: continue
    if "wandb" in args: continue
    args = args.strip().replace("\\","")
    print(args)
    k, v = args.split()
    total_args[k] = dict(value=v)
print(total_args)

prefix = r"""
command:
  - ${env}
  - ${interpreter}
  - "main.py"
  - "--wandb"
  - "--overwrite_cache"
  - ${args}
program: main.py
method: grid
name: fb15k-knn-test
metric:
  name: Test/mrr
  goal: maxmize

"""
with open(output_path, 'w') as file:
    file.write(prefix)
    yaml.dump(dict(parameters=total_args), file)
