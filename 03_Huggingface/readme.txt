conda create -n huggingface python=3.10

conda activate huggingface

pip install -r requirements.txt

pip install torch==2.3.1+cu121 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
清华源无cuda版本

huggingface-cli download Qwen/Qwen2.5-7B-Instruct --max-workers 1 --local-dir-use-symlinks false --local-dir XXX

modelscope download Qwen/Qwen2.5-7B-Instruct --max-workers 1 --local_dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75

#pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/