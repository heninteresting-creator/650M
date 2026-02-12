# main.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import csv
import time
from data_registry import get_strategy
from prompt_factory import build_prompt_messages
from api_engine import batch_inference

# ================= 🎛️ 控制面板 (Control Panel) =================

# 1. 决定用哪个数据集 (支持 'gsm8k' 或 'math')
TARGET_DATASET = "gsm8k" 

# 2. 决定用哪个模型
TARGET_MODEL = "Qwen/Qwen3-8B"

# 3. 决定跑多少条数据
SAMPLE_SIZE = 1000  

# 4. 定义你要跑哪些实验 (四组实验矩阵)
EXPERIMENT_MATRIX = [
    {"label": "exp_01_no_cot_no_shot",   "cot": False, "shot": False},
    {"label": "exp_02_with_cot_no_shot", "cot": True,  "shot": False},
    {"label": "exp_03_no_cot_few_shot",  "cot": False, "shot": True},
    {"label": "exp_04_with_cot_few_shot","cot": True,  "shot": True},
]

# ==============================================================

def main():
    print(f"数据集: {TARGET_DATASET} | 模型: {TARGET_MODEL}")
    strategy = get_strategy(TARGET_DATASET)
    test_ds = strategy.load_data(split="test", limit=SAMPLE_SIZE)
    full_ds_for_shots = strategy.load_data(split="test") 
    shot_data = full_ds_for_shots.select(range(len(full_ds_for_shots)-4, len(full_ds_for_shots)))
    
    for exp in EXPERIMENT_MATRIX:
        label = exp["label"]
        use_cot = exp["cot"]
        use_shot = exp["shot"]
        filename = f"result_{TARGET_DATASET}_{label}.csv"
        
        # ==================== 新增：断点加载逻辑 ====================
        existing_results = []
        if os.path.exists(filename):
            print(f"🔍 发现已有文件 {filename}，正在读取进度...")
            with open(filename, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                existing_results = [row for row in reader]
        
        print(f"\n>>> 正在运行: {label} (CoT={use_cot}, Shot={use_shot})")
        
        # 将进度传给 batch_inference
        results = batch_inference(
            dataset=test_ds,
            strategy=strategy,
            prompt_builder_func=build_prompt_messages,
            model_name=TARGET_MODEL,
            use_cot=use_cot,
            use_shot=use_shot,
            shot_data=shot_data,
            existing_results=existing_results
        )
        
        # 写入结果 (直接覆盖旧文件，因为 results 已经包含了旧数据和新数据)
        headers = ["question", "ground_truth", "model_output", "extracted", "tokens", "is_correct", "error"]
        with open(filename, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
            
        print(f"✅ {label} 处理完成，当前有效数据：{len([r for r in results if not r['error']])} 条")
        time.sleep(2)

if __name__ == "__main__":
    main()