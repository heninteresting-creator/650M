import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import csv
import argparse
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--api_key', type=str, default="sk-qfaatcoxrfhoagnwtgjikyudjnxoytteombhqsooxbtxhpnl")
    parser.add_argument('--samples', type=int, default=250)
    parser.add_argument('--workers', type=int, default=10)
    return parser.parse_args()

def extract_number(text):
    if not text: return ""
    match = re.search(r"####\s*(-?[\d\.,]+)", str(text))
    if match: return match.group(1).replace(",", "").rstrip('.')
    nums = re.findall(r"-?[\d\.]+", str(text))
    return nums[-1].rstrip('.') if nums else ""

def query_api(client, question, use_cot, model_name):
    if use_cot:
        sys_msg = "你是一个数学专家。请逐步推理，并在最后给出格式为 'The answer is #### [数字]' 的结论。"
        user_msg = f"Question: {question}\nAnswer:"
        max_tokens = 2048
    else:
        sys_msg = "你是一个数学专家。请直接给出最终数值答案，格式为 'The answer is #### [数字]'，不要解释过程。"
        user_msg = f"Question: {question}\nDirect Answer: The answer is ####"
        max_tokens = 2048
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=max_tokens,
            temperature=0.01,
            timeout=30.0
        )
        
        output = response.choices[0].message.content
        tokens = response.usage.completion_tokens
        return output, tokens, ""
    except Exception as e:
        return "", 0, str(e)

def process_item(item, client, args, use_cot):
    question = item['question']
    gt = extract_number(item['answer'])
    
    output, tokens, error = query_api(client, question, use_cot, args.model)
    extracted = extract_number(output)
    
    # 判断是否正确
    is_correct = "Wrong"
    if extracted and gt:
        try:
            if abs(float(extracted) - float(gt)) < 1e-5:
                is_correct = "Correct"
        except:
            pass
    
    return {
        "question": question,
        "ground_truth": gt,
        "model_output": output,
        "model_answer": extracted,
        "is_correct": is_correct,
        "error": error,
        "tokens": tokens
    }

def main():
    args = parse_args()
    print(f"使用模型: {args.model}")
    print(f"样本数: {args.samples}")
    
    client = OpenAI(api_key=args.api_key, base_url="https://api.siliconflow.cn/v1")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{args.samples}]")
    
    # 运行两种模式
    for use_cot, label in [(False, "without_cot"), (True, "with_cot")]:
        print(f"\n运行 {label}...")
        filename = f"results_{label}.csv"
        
        with open(filename, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "ground_truth", "model_output", 
                "model_answer", "is_correct", "error", "tokens"
            ])
            writer.writeheader()
            
            # 多线程处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(process_item, item, client, args, use_cot) 
                    for item in dataset
                ]
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(dataset), desc=label):
                    writer.writerow(future.result())
                    f.flush()
        
        print(f"结果已保存到 {filename}")

if __name__ == "__main__":
    main()