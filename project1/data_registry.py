# data_registry.py
import re
from datasets import load_dataset

class DataStrategy:
    def __init__(self, dataset_name):
        self.name = dataset_name

    def load_data(self, split="test", limit=None):
        """统一加载逻辑，返回标准化的 list"""
        print(f"Loading dataset: {self.name}...")
        
        if self.name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split=split)
            # 统一字段名：question, answer
        elif self.name == "math":
            ds = load_dataset("hendrycks/competition_math", split=split)
            # MATH 的字段是 'problem' 和 'solution'，这里做映射
            ds = ds.map(lambda x: {"question": x["problem"], "answer": x["solution"]})
        else:
            raise ValueError(f"暂不支持数据集: {self.name}")

        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    def extract_ground_truth(self, text):
        """从文本中提取 GSM8K 格式的答案（#### 1234）"""
        text = str(text)
        
        # GSM8K 格式: #### 1234
        match = re.search(r"####\s*(-?[\d\.,]+)", text)
        if match: 
            return match.group(1).replace(",", "").rstrip('.')
        # 兜底：找最后一个数字
        nums = re.findall(r"-?[\d\.]+", text)
        return nums[-1].rstrip('.') if nums else ""

    def clean_shot_answer(self, raw_answer):
        """为 Without CoT 模式清洗 Few-shot 里的推理过程"""
        val = self.extract_ground_truth(raw_answer)
        if self.name == "gsm8k":
            return f"The answer is #### {val}"
        elif self.name == "math":
            return f"The answer is \\boxed{{{val}}}"
        return val

# 工厂函数
def get_strategy(dataset_name):
    return DataStrategy(dataset_name)