import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import csv
import torch
import gc
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

model_id = "Qwen/Qwen3-0.6B" 
output_file = "qwen_gsm8k_stream_results.csv"
BATCH_SIZE = 16

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

clear_gpu_memory()

def extract_answer_number(text):
    
    if not text:
        return ""

    text = text.replace(",", "")

    match = re.search(r"####\s*(-?[\d\.]+)", text)
    if match:
        return match.group(1)

    numbers = re.findall(r"-?[\d\.]+", text)
    if numbers:
        return numbers[-1]
    
    return ""

# 初始化 Pipeline
print(f"正在加载模型 {model_id}...")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto", 
    device_map="auto",
)

# 设置Padding
if pipe.tokenizer.pad_token_id is None:
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
pipe.tokenizer.padding_side = "left"

# 加载数据集
print("正在读取 GSM8K 测试集...")
dataset = load_dataset("openai/gsm8k", "main", split="test")
total_samples = len(dataset) 

# 生成器
def prompt_generator(data_source):
    for item in data_source:
        messages = [
            {"role": "system", "content": "你是一个数学专家。请逐步推理，并在最后给出格式为 'The answer is #### [数字]' 的结论。"},
            {"role": "user", "content": item["question"]}
        ]
        yield messages

# 开始inference并写入 CSV
print(f"开始流式解题 (Batch Size = {BATCH_SIZE})...")

inference_results = pipe(
    prompt_generator(dataset),
    batch_size=BATCH_SIZE,
    max_new_tokens=1024,
    do_sample=False, 
    repetition_penalty=1.05, 
    pad_token_id=pipe.tokenizer.pad_token_id
)

with open(output_file, mode="w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["问题", "标准过程", "标准答案", "模型输出", "模型答案", "状态"])

    correct_count = 0
    
    for original_item, model_out in tqdm(zip(dataset, inference_results), total=total_samples):
        # 提取标准信息
        question = original_item["question"]
        gt_process = original_item["answer"]
        gt_answer = extract_answer_number(gt_process)
        
        # 提取模型信息
        try:
            model_output = model_out[0]["generated_text"][-1]["content"]
            model_answer = extract_answer_number(model_output)
            
            try:
                is_correct = float(gt_answer) == float(model_answer)
            except:
                is_correct = (gt_answer == model_answer)
                
            status = "正确" if is_correct else "错误"
            if is_correct: correct_count += 1
            
        except Exception as e:
            model_output = f"ERROR: {str(e)}"
            model_answer = ""
            status = "解析失败"

        # 写入行
        writer.writerow([
            question,
            gt_process,
            gt_answer,
            model_output.strip(),
            model_answer,
            status
        ])
        
        f.flush()

# 打印结果
accuracy = (correct_count / total_samples) * 100
print(f"\n解题完成")
print(f"总样本数: {total_samples}")
print(f"正确数量: {correct_count}")
print(f"准确率: {accuracy:.2f}%")
print(f"结果: {output_file}")

clear_gpu_memory()