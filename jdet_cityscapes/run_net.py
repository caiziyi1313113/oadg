import argparse
import os
import sys

import jittor as jt
# Managed allocator can be unstable in some WSL/driver setups; keep it off.
jt.flags.use_cuda_managed_allocator = 0

# Make local modules importable
# 把“本地目录”加入模块搜索路径（能 import 自己写的模块）
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# 把 JDet 的 python 路径加入 sys.path（如果存在）
# Optional: add JDet python path if it exists
# 将 D:\ 替换为 /mnt/d/，并将反斜杠 \ 替换为正斜杠 /
# D:\sim2real\OA-DG\JDet
DEFAULT_JDET_PY = r"D:\sim2real\OA-DG\JDet\python" if os.name == "nt" else "/mnt/d/sim2real/OA-DG/JDet/python"

if os.path.isdir(DEFAULT_JDET_PY) and DEFAULT_JDET_PY not in sys.path:
    sys.path.insert(0, DEFAULT_JDET_PY)

# Register custom dataset
# “注册”自定义数据集和模型
import datasets.repeat_dataset  # noqa: F401
import datasets.resize_by_img_scale  # noqa: F401
import datasets.cityscapes_dataset  # noqa: F401
import datasets.oa_mix  # noqa: F401
import models.hbb_head  # noqa: F401
import models.faster_rcnn_hbb  # noqa: F401
import models.faster_rcnn_hbb_multi  # noqa: F401
import models.fdd_modules  # noqa: F401
import models.faster_rcnn_hbb_fdd  # noqa: F401
import models.param_groups  # noqa: F401
import models.oadg_losses  # noqa: F401

# 导入训练入口：Runner 和配置初始
# 你的训练/验证/测试的控制器（封装了 run/val/test 等流程）
from runner import Runner
# 用配置文件初始化全局配置（JDet/Jittor detection 常见做法）
from jdet.config import init_cfg, get_cfg

# 参数解析
def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    # 配置文件路径；不传就不初始化配置（可能 Runner 内部有默认配置或自己处理）
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test,vis_test",
        type=str,
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
    )

    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    # 从命令行解析参数
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda = 1

    assert args.task in ["train", "val", "test", "vis_test"], (
        f"{args.task} not support, please choose [train,val,test,vis_test]"
    )
    # 如果给了配置文件，就初始化配置
    if args.config_file:
        init_cfg(args.config_file)
        cfg = get_cfg()
        if getattr(cfg, "pretrained_weights", None) == "":
            cfg.pretrained_weights = None
    # 创建 Runner 实例
    runner = Runner()
    # 根据任务类型调用 Runner 的对应方法
    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()
    elif args.task == "vis_test":
        runner.run_on_images(args.save_dir)


if __name__ == "__main__":
    main()
