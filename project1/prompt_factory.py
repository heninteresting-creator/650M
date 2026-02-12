# prompt_factory.py

def build_prompt_messages(item, use_cot, use_shot, shot_data, strategy):
    question = item['question']
    
    # 1. 构建 Few-shot 文本
    context = ""
    if use_shot and shot_data:
        context = "Here are some examples:\n\n"
        for s in shot_data:
            q_s = s['question']
            if use_cot:
                # CoT
                a_s = s['answer']
            else:
                # No CoT: 调用 strategy 清洗，只留答案
                a_s = strategy.clean_shot_answer(s['answer'])
            
            context += f"Question: {q_s}\nAnswer: {a_s}\n\n"

    # 2. 构建 System Prompt 和 User Input
    if use_cot:
        sys_msg = "You are a math expert. Reason step by step, then end with 'The answer is #### [number]'." #格式可能需要根据数据集微调
        user_msg = context + f"Question: {question}\nAnswer:"
    else:
        sys_msg = "You are a math expert. Direct answer only. Do NOT explain."
        user_msg = context + f"Question: {question}\nDirect Answer: The answer is"

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]