import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import csv
import re
import argparse
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import torch

# ================= 1. 配置参数 =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='api', choices=['local', 'api'])
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen3-8B") 
    parser.add_argument('--api_key', type=str, default="sk-qfaatcoxrfhoagnwtgjikyudjnxoytteombhqsooxbtxhpnl")
    parser.add_argument('--samples', type=int, default=250)
    parser.add_argument('--workers', type=int, default=10)
    return parser.parse_args()

# ================= 2. 核心提取工具 =================
def extract_val(text):
    if not text: return ""
    # 匹配 GSM8K 标准格式 #### 数字
    match = re.search(r"####\s*(-?[\d\.,]+)", str(text))
    if match: return match.group(1).replace(",", "").rstrip('.')
    # 兜底：找最后一个出现的数字
    nums = re.findall(r"-?[\d\.]+", str(text))
    return nums[-1].rstrip('.') if nums else ""

def get_shots(full_dataset, use_cot):
    # 取测试集最后3条作为 Few-shot 示例
    shot_data = full_dataset.select(range(len(full_dataset)-3, len(full_dataset)))
    prompt = "Here are some examples:\n\n"
    for s in shot_data:
        ans = s['answer'] if use_cot else f"The answer is #### {extract_val(s['answer'])}"
        prompt += f"Question: {s['question']}\nAnswer: {ans}\n\n"
    return prompt

# ================= 3. 推理核心逻辑 =================
def run_single_task(item, use_cot, shots, args, client=None):
    question = item['question']
    gt_val = extract_val(item['answer'])
    
    if use_cot:
        sys_prompt = "You are a math expert. Reason step by step, then end with 'The answer is #### [number]'."
        user_input = shots + f"Question: {question}\nAnswer:"
        max_tok = 2048
    else:
        sys_prompt = "You are a math expert. Direct answer only. Format: 'The answer is #### [number]'."
        user_input = shots + f"Question: {question}\nDirect Answer: The answer is ####"
        max_tok = 2048

    # 预设结果字典，显式包含 is_correct
    res = {
        "question": question,
        "ground_truth": gt_val,
        "model_output": "",
        "extracted": "",
        "tokens": 0,
        "is_correct": "Wrong", # 默认为错
        "error": ""
    }

    try:
        response = client.chat.completions.create(
            model=args.model_path,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_input}],
            max_tokens=max_tok, 
            temperature=0.01, 
            timeout=60.0 
        )
        raw = response.choices[0].message.content
        res['tokens'] = response.usage.completion_tokens
        res['model_output'] = f"The answer is #### {raw}" if not use_cot else raw
        
        # 提取并判定
        pred_val = extract_val(res['model_output'])
        res['extracted'] = pred_val
        
        # 数值比对
        try:
            if abs(float(pred_val) - float(gt_val)) < 1e-5:
                res['is_correct'] = "Correct"
        except:
            res['is_correct'] = "Wrong"

    except Exception as e:
        res['error'] = str(e)
    return res

# ================= 4. 主程序 =================
def main():
    args = parse_args()
    print(f"--- 任务启动: {args.model_path} ---")
    
    client = OpenAI(api_key=args.api_key, base_url="https://api.siliconflow.cn/v1")

    # 加载数据集
    full_ds = load_dataset("openai/gsm8k", "main", split="test")
    dataset = full_ds.select(range(args.samples))

    # 定义csv文件的列名
    fieldnames = ["question", "ground_truth", "model_output", "extracted", "tokens", "is_correct", "error"]

    for use_cot in [False, True]:
        label = "with_cot" if use_cot else "without_cot"
        filename = f"data_api_{label}.csv"
        shots = get_shots(full_ds, use_cot)
        
        print(f"\n正在执行: {label} ...")
        
        with open(filename, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 使用多线程提高速度
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_single_task, item, use_cot, shots, args, client) for item in dataset]
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc=label):
                    row = future.result()
                    writer.writerow(row)
                    f.flush()

if __name__ == "__main__":
    main()