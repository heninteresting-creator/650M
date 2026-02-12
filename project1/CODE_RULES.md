# 代码规则文档

本文档描述了 `project1` 目录下的代码规范和架构规则。

## 📁 项目结构

```
project1/
├── config.py              # 配置文件（API密钥、并发设置）
├── data_registry.py       # 数据加载和答案提取策略
├── prompt_factory.py      # 提示词构建工厂
├── api_engine.py          # API调用引擎（重试、批量推理）
├── main.py                # 主程序入口
└── final_analyze_all.py   # 结果分析和可视化
```

## 🏗️ 架构设计原则

### 1. 模块职责分离

- **config.py**: 集中管理所有配置项，包括 API 密钥、并发数、超时时间
- **data_registry.py**: 负责数据集加载、答案提取逻辑，使用策略模式
- **prompt_factory.py**: 专门负责构建不同场景下的提示词
- **api_engine.py**: 封装 API 调用逻辑，包括重试机制和并发控制
- **main.py**: 作为控制面板，定义实验矩阵，协调各模块执行
- **final_analyze_all.py**: 独立的结果分析和可视化模块

### 2. 策略模式 (Strategy Pattern)

使用 `DataStrategy` 类封装不同数据集的处理逻辑：

```python
class DataStrategy:
    def load_data(self, split="test", limit=None)      # 数据加载
    def extract_ground_truth(self, text)                # 答案提取
    def clean_shot_answer(self, raw_answer)            # Few-shot 清洗
```

**规则**：
- 每个数据集对应一个策略实例
- 通过工厂函数 `get_strategy(dataset_name)` 创建策略
- 策略方法必须返回标准化格式

### 3. 配置集中管理

**规则**：
- 所有配置项必须在 `config.py` 中定义
- 使用大写常量命名：`API_KEY`, `BASE_URL`, `MAX_WORKERS`, `TIMEOUT`
- 敏感信息（如 API 密钥）应使用环境变量或配置文件，避免硬编码

## 📝 代码风格规范

### 1. 注释规范

- **文件头注释**：使用 `# filename.py` 格式
- **函数文档字符串**：使用中文三引号文档字符串
- **关键逻辑注释**：使用中文注释说明复杂逻辑

示例：
```python
def extract_ground_truth(self, text):
    """从文本中提取 GSM8K 格式的答案（#### 1234）"""
    text = str(text)
    
    # GSM8K格式
    match = re.search(r"####\s*(-?[\d\.,]+)", text)
```

### 2. 命名规范

- **变量名**：使用小写字母和下划线，如 `use_cot`, `shot_data`
- **常量名**：使用大写字母和下划线，如 `MAX_WORKERS`, `TARGET_MODEL`
- **函数名**：使用小写字母和下划线，如 `build_prompt_messages`, `batch_inference`
- **类名**：使用驼峰命名，如 `DataStrategy`

### 3. 函数设计规则

- **单一职责**：每个函数只做一件事
- **参数明确**：函数参数要有清晰的类型和用途
- **返回值统一**：相同功能的函数返回格式应保持一致

示例：
```python
def run_single(item, messages, strategy, model_name, max_tok):
    """处理单个题目的推理"""
    # 返回统一格式的字典
    return {
        "question": item['question'],
        "ground_truth": gt,
        "model_output": content,
        "extracted": pred,
        "tokens": tokens,
        "is_correct": is_correct,
        "error": ""
    }
```

## 🔄 错误处理和重试机制

### 1. 指数退避重试

**规则**：
- 最大重试次数：5 次
- RateLimitError (429) 特殊处理：基础等待 5 秒 + 指数增长 + 随机抖动
- 其他异常：固定等待 2 秒后重试

```python
max_retries = 5
for i in range(max_retries):
    try:
        # API 调用
    except RateLimitError as e:
        wait_time = (2 ** i) + random.random() * 2 + 5
        time.sleep(wait_time)
    except Exception as e:
        if i < max_retries - 1:
            time.sleep(2)
            continue
```

### 2. 错误信息记录

**规则**：
- 所有错误必须记录在返回结果的 `error` 字段中
- 即使失败也要返回完整的字典结构，避免后续处理崩溃

## 🚀 并发控制规则

### 1. 线程池配置

- 使用 `ThreadPoolExecutor` 进行并发控制
- 并发数由 `config.MAX_WORKERS` 配置
- 提交任务时添加延迟，避免瞬间并发冲垮 TPM

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
    for item in todo_dataset:
        futures.append(executor.submit(run_single, ...))
        if config.MAX_WORKERS > 1: 
            time.sleep(0.1)  # 避免瞬间并发
```

### 2. 进度显示

- 使用 `tqdm` 显示处理进度
- 进度条应显示总任务数和当前进度

## 💾 数据持久化规则

### 1. CSV 文件格式

**标准字段**：
- `question`: 问题文本
- `ground_truth`: 标准答案
- `model_output`: 模型完整输出
- `extracted`: 提取的答案
- `tokens`: 消耗的 token 数
- `is_correct`: 是否正确（"Correct" 或 "Wrong"）
- `error`: 错误信息（如果有）

### 2. 文件编码

- 统一使用 `utf-8-sig` 编码，确保 Excel 能正确打开

### 3. 断点续传

**规则**：
- 支持从已有 CSV 文件恢复进度
- 判断已完成题目的标准：`error` 为空且 `is_correct` 有值
- 自动过滤已完成的题目，只处理未完成的

```python
done_questions = {res['question'] for res in results if not res.get('error')}
todo_dataset = [item for item in dataset if item['question'] not in done_questions]
```

## 🧪 实验配置规则

### 1. 实验矩阵定义

在 `main.py` 中使用 `EXPERIMENT_MATRIX` 定义所有实验配置：

```python
EXPERIMENT_MATRIX = [
    {"label": "exp_01_no_cot_no_shot",   "cot": False, "shot": False},
    {"label": "exp_02_with_cot_no_shot",  "cot": True,  "shot": False},
    {"label": "exp_03_no_cot_few_shot",  "cot": False, "shot": True},
    {"label": "exp_04_with_cot_few_shot","cot": True,  "shot": True},
]
```

### 2. 文件命名规范

- 格式：`result_{dataset}_{label}.csv`
- 示例：`result_gsm8k_exp_01_no_cot_no_shot.csv`

### 3. 控制面板

在 `main.py` 顶部定义所有可配置项：
- `TARGET_DATASET`: 目标数据集
- `TARGET_MODEL`: 目标模型
- `SAMPLE_SIZE`: 样本数量

## 📊 答案提取规则

### 1. GSM8K 格式

- 标准格式：`#### 1234`
- 提取逻辑：
  1. 优先匹配 `####` 后的数字
  2. 移除千位分隔符（逗号）
  3. 移除末尾的点
  4. 兜底：如果没找到 `####`，提取文本中最后一个数字

### 2. 答案比较规则

- 数值比较：使用 `abs(float(pred) - float(gt)) < 1e-5` 判断
- 字符串比较：作为兜底方案，精确匹配字符串

```python
try:
    if abs(float(pred) - float(gt)) < 1e-5:
        is_correct = "Correct"
except:
    if str(pred).strip() == str(gt).strip() and pred:
        is_correct = "Correct"
```

## 🎨 提示词构建规则

### 1. CoT vs Direct 模式

- **CoT 模式**：要求模型逐步推理，最后给出答案
- **Direct 模式**：要求直接给出答案，不解释

### 2. Few-shot 处理

- **With CoT**：Few-shot 示例包含完整推理过程
- **Without CoT**：Few-shot 示例只包含答案，使用 `clean_shot_answer` 清洗

### 3. 消息格式

统一使用 OpenAI Chat API 格式：
```python
[
    {"role": "system", "content": sys_msg},
    {"role": "user", "content": user_msg}
]
```

## 📈 Token 管理规则

### 1. 最大 Token 配置

- **CoT 模式**：`max_tokens = 1024`
- **Direct 模式**：`max_tokens = 128`

### 2. Token 统计

- 记录每次 API 调用的 `total_tokens`
- 用于后续的成本分析和效率评估

## 🔍 代码质量要求

### 1. 异常处理

- 所有可能失败的操作都要有 try-except
- 不要使用裸露的 `except:`，至少指定异常类型
- 记录详细的错误信息

### 2. 代码复用

- 避免重复代码
- 公共逻辑提取为函数
- 使用工厂模式创建对象

### 3. 可维护性

- 配置项集中管理
- 使用常量而非魔法数字
- 函数职责清晰，便于测试和修改

## 📌 注意事项

1. **API 密钥安全**：当前 `config.py` 中硬编码了 API 密钥，生产环境应使用环境变量
2. **并发控制**：根据 API 提供商的 TPM 限制调整 `MAX_WORKERS`
3. **超时设置**：根据模型响应时间调整 `TIMEOUT` 值
4. **数据验证**：在关键步骤添加数据验证，确保数据格式正确
5. **日志记录**：建议添加日志系统，记录关键操作和错误信息

## 🔄 未来改进建议

1. 使用环境变量管理敏感配置
2. 添加日志系统（如 `logging` 模块）
3. 添加单元测试
4. 支持更多数据集类型
5. 添加配置文件验证
6. 改进错误处理和重试策略的可配置性

