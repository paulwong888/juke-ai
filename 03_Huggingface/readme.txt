conda create -n huggingface python=3.10

conda activate huggingface

pip install -r requirements.txt

pip install torch==2.3.1+cu121 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
清华源无cuda版本

#pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/