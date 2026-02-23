## 目录
### 目录结构

```text
├── JDet/ 
├── data/                    # 数据集目录 
│   ├── cityscapes/          # Cityscapes 数据集文件
│   ├── S-DGOD/              # S-DGOD 数据集文件
│   └── external/            # 其他外部数据
├── jdet_cityscapes/         # 核心源代码包
│   ├── configs/             # 各种模型配置文件
│   ├── datasets/            # 数据集加载逻辑
│   ├── models/              # 模型定义 
│   ├── tools/               # 辅助工具脚本 
│   ├── runner.py            # 训练/测试的核心调度器逻辑
│   ├── run_net.py           # 项目主入口 
│   ├── work_dirs/           # 训练实验结果 
│   └── minus_jdet.pkl       # 预训练权重文件
└──readme.md



### 1.1 建议用独立虚拟环境

```bash
# 建议新建一个专用环境
conda create -n jdet python=3.10 -y
conda activate jdet
```
```bash
# 基础构建工具（仍然建议装系统包）
sudo apt update
sudo apt install -y build-essential git cmake ninja-build python3-dev
```
### 1.2 安装 Jittor 并做 GPU 验证
```bash
pip install jittor
```

### 1.3 安装cuDNN
cuDNN（tar安装）cudnn-dev（建议使用tar文件安装，[参考链接](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)）

conda 的 cuDNN：对 PyTorch 友好  
Jittor 的 cuDNN：必须是系统级（tar 包）

不能通过conda 安装，那个是针对pytorch使用，计图需要手动安装到可识别路径

```bash
# 先清理之前的缓存
rm -rf ~/.cache/jittor

# 让 Jittor 自动配置 CUDA/cuDNN 环境
python3.10 -m jittor_utils.install_cuda
```
### 1.4 安装 JDet

```bash
git clone https://github.com/Jittor/JDet
cd JDet
python -m pip install -r requirements.txt
```

```bash
cd JDet
# suggest this 
python setup.py develop
# or
python setup.py install
```

### 1.5 验证

验收（能跑一个最小 config 的 train 或至少 import 成功）：

```bash
python -c "import jdet; print('jdet import ok')"
```



## 数据集

- [城市景观](https://www.cityscapes-dataset.com/)：一个包含 50 个城市的城市街景的数据集，并附有详细的标注。
- [多样化天气数据集](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)：该数据集包含各种天气条件，可用于对模型进行稳健的测试和开发，这对于自动驾驶应用至关重要。

## 训练

脚本示例
```bash
python run_net.py --config-file configs/faster_rcnn_r50_fpn_1x_cityscapes_coco_jdet.py --task train
```
## 验证

脚本示例
```bash
python jdet_cityscapes/run_net.py \
  --config-file jdet_cityscapes/configs/faster_rcnn_r50_fpn_cityscapes_jittor_oadg.py \
  --task train
```

## 生成离线的cityscape-c数据集
jittor
```bash
python jdet_cityscapes/tools/analysis_tools/get_corrupted_dataset.py \
  jdet_cityscapes/configs/faster_rcnn_r50_fpn_1x_cityscapes_coco_jdet.py \
  --show-dir /mnt/d/sim2real/OA-DG/ws/data/cityscapes-c \
  --corruptions benchmark
```
mmdet
```bash
python3 tools/analysis_tools/get_corrupted_dataset.py \
  configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_oadg.py \
  --show-dir /root/autodl-tmp/sim2real/OA-DG/ws/data/cityscapes-c \
  --corruptions benchmark \
  --cfg-options \
    data.test.ann_file=/root/autodl-tmp/sim2real/OA-DG/ws/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json \
    data.test.img_prefix=/root/autodl-tmp/sim2real/OA-DG/ws/data/cityscapes/leftImg8bit/val/ \
    data.workers_per_gpu=2
```

## 测试

生成鲁棒性结果 示例

```bash
python jdet_cityscapes/tools/analysis_tools/test_robustness.py \
  jdet_cityscapes/configs/faster_rcnn_r50_fpn_1x_cityscapes_coco_jdet.py \
  jdet_cityscapes/work_dirs/faster_rcnn_r50_fpn_cityscapes_coco_jdet/checkpoints/ckpt_16.pkl \
  --out jdet_cityscapes/work_dirs/faster_rcnn_r50_fpn_cityscapes_coco_jdet/robustness_last.json \
  --corruptions benchmark \
  --load-dataset corrupted
```


统计 mPC 示例

```bash
python jdet_cityscapes/tools/compute_cityscapes_c_metrics.py \
  jdet_cityscapes/work_dirs/faster_rcnn_r50_fpn_cityscapes_coco_jdet_oadg_oamix/robustness_last.json \
  --output jdet_cityscapes/work_dirs/faster_rcnn_r50_fpn_cityscapes_coco_jdet_oadg_oamix/robustness_last_mpc.json
```




