# MultiModalityDet

## Install

```bash
conda create -n mmd python=3.8 -y
conda activate mmd
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install black tqdm -y  (Optional)
pip install openmim==0.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine==0.8.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet3d==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
