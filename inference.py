import os
# 1. 解决之前遇到的 DLL 冲突问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 2. 设置 Hugging Face 国内镜像源，确保下载不卡顿
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

def run_inference():
    # 设定你要跑的 650M 模型 ID
    model_id = "facebook/esm2_t33_650M_UR50D"
    
    print(f"正在从镜像站加载模型: {model_id}...")
    print("提示：第一次运行会自动下载约 2.4GB 的权重文件，请耐心等待。")

    # 3. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 使用 .half() 可以将模型转为 16位精度，能大幅减少显存占用并加快速度
    model = AutoModelForMaskedLM.from_pretrained(model_id).half()

    # 4. 自动检测设备 (有显卡用 CUDA，没显卡用 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"模型已成功加载到设备: {device}")

    # 5. 准备输入数据 (一段蛋白质氨基酸序列)
    # 我们故意放一个 <mask> 看看模型能否预测出缺失的部分
    sequence = "MAPLRKTYVLKRAEQMTREEVEKYLKEGQIDLVK<mask>YV"
    inputs = tokenizer(sequence, return_tensors="pt").to(device)

    # 6. 执行推理
    print("开始推理...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 7. 简单处理结果
    # 拿到 logits 后，我们看看模型预测的形状
    result_shape = outputs.logits.shape
    print("-" * 30)
    print(f"推理成功！输出张量的形状为: {result_shape}")
    print("这意味着模型已经根据你的输入计算出了每一个位置的特征向量。")
    print("-" * 30)

if __name__ == "__main__":
    run_inference()