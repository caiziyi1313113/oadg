import torch
import pickle

def ultra_minimalist_convert(pt_path, out_path):
    # 1. 直接读取 PyTorch 权重
    state_dict = torch.load(pt_path, map_location='cpu')['state_dict']
    
    # 2. 转换为 Numpy 格式字典
    # 注意：JDet 的加载器在发现 shape 不匹配时会打出 Warning 并跳过该层
    jdet_weights = {k: v.numpy() for k, v in state_dict.items()}

    # 3. 保存为 pkl
    with open(out_path, 'wb') as f:
        pickle.dump(jdet_weights, f)
    print(f"转换完成，已生成: {out_path}")

# 执行
ultra_minimalist_convert('./premodel/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118 (1).pth', './minimal_jdet.pkl')