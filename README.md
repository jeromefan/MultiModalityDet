# MultiModalityDet

## 环境配置

```bash
conda create -n mmd python=3.8 -y
conda activate mmd
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install black tqdm -y  # 可选
pip install openmim==0.3.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmengine==0.8.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmcv==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet==3.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install mmdet3d==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install yapf==0.40.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

git clone https://github.com/jeromefan/MultiModalityDet.git
cd MultiModalityDet
python projects/BEVFusion/setup.py develop
```

## 预训练权重

请到 checkpoints/ckpt.md 查询谷歌云盘下载链接。

## 检测Demo - 使用 BEVFsion 对Carla生成的数据进行多模态检测

```bash
python multi_modality_data_generation.py  # 可选，data/carla 文件夹下已放了一个生成的demo数据
python mul_modality_det.py
```

## 攻击

```bash
python fgsm_img_attack.py
```
