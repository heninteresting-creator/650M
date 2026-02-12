# api_engine.py
import concurrent.futures
import time
import random
from openai import OpenAI, RateLimitError  
from tqdm import tqdm
import config

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

def run_single(item, messages, strategy, model_name, max_tok):
    # ====================指数退避重试逻辑====================
    max_retries = 5
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.01,
                max_tokens=max_tok,
                timeout=config.TIMEOUT
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            pred = strategy.extract_ground_truth(content)
            gt = strategy.extract_ground_truth(item['answer'])
            
            is_correct = "Wrong"
            try:
                if abs(float(pred) - float(gt)) < 1e-5:
                    is_correct = "Correct"
            except:
                if str(pred).strip() == str(gt).strip() and pred:
                    is_correct = "Correct"

            return {
                "question": item['question'],
                "ground_truth": gt,
                "model_output": content,
                "extracted": pred,
                "tokens": tokens,
                "is_correct": is_correct,
                "error": ""
            }

        except RateLimitError as e:
            # 针对 429 错误的特殊处理
            wait_time = (2 ** i) + random.random() * 2 + 5 # 基础等待5秒+指数增加
            print(f"\nRate Limit (429) hit. Waiting {wait_time:.1f}s before retry...")
            time.sleep(wait_time)
            continue
        except Exception as e:
            # 其他网络或 API 错误
            if i < max_retries - 1:
                time.sleep(2)
                continue
            return {
                "question": item['question'], "ground_truth": "", "model_output": "", 
                "extracted": "", "tokens": 0, "is_correct": "Wrong", "error": str(e)
            }
    return {"question": item['question'], "is_correct": "Wrong", "error": "Max retries exceeded"}

def batch_inference(dataset, strategy, prompt_builder_func, model_name, use_cot, use_shot, shot_data, existing_results=None):
    results = []
    # 如果有历史结果，先放进结果列表
    if existing_results:
        results.extend(existing_results)
    
    max_tok = 1024 if use_cot else 128
    
    # 过滤掉已经成功跑完的题目 (判断标准：error为空且is_correct有值)
    done_questions = {res['question'] for res in results if not res.get('error')}
    todo_dataset = [item for item in dataset if item['question'] not in done_questions]
    
    if not todo_dataset:
        print("所有题目已完成，无需重跑。")
        return results

    print(f"待补全题目数量: {len(todo_dataset)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = []
        for item in todo_dataset:
            msgs = prompt_builder_func(item, use_cot, use_shot, shot_data, strategy)
            futures.append(executor.submit(run_single, item, msgs, strategy, model_name, max_tok))
            # 等待一下再提交，避免瞬间并发冲垮 TPM
            if config.MAX_WORKERS > 1: time.sleep(0.1)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(todo_dataset), desc="Processing"):
            results.append(future.result())
            
    return results