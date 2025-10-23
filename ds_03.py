import json
import os
import pathlib
import pickle
import re
import sys

from openai import OpenAI

# 准确率87%

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"

# 修复：确保缓存目录存在
PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

with open(BASE_PATH / 'deepseek.key', 'r') as f:
    API_KEY = f.read().strip()

MODEL_NAME = "deepseek-chat"
PROMPT_VERSION = "v003"  # 使用新版本的提示词

if "-mini" in MODEL_NAME:
    ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/deepseek-mini_classified.json")
else:
    ALL_CLASSES_JSON = str(BASE_PATH / "data/Classes/deepseek_classified.json")

sys.path.append(str(CODE_BASE_PATH))

from util.dataset import sample_data
from classifierutil import get_gold_label, ALLOWED_CLASSES
from report_metrics import report_metrics

EXAMPLE_SIZE = 5
EXAMPLE_SIZE_TYPE = "fixed"
DS_PATHS = [
    str(BASE_PATH / "data/Datasets/DataXFormer"),
    str(BASE_PATH / "data/Datasets/AutoJoin"),
    str(BASE_PATH / "data/Datasets/FlashFill"),
    str(BASE_PATH / "data/Datasets/All_TDE"),
]


def is_scientific_notation_conversion(examples):
    """
    判断是否为科学计数法转换
    规则：源数据是常规数字，目标数据是科学计数法格式
    """
    scientific_notation_pattern = r'^-?\d+(\.\d+)?[eE][+-]?\d+$'

    for exp in examples:
        source = str(exp[0])
        target = str(exp[1])

        # 检查源数据是否为常规数字
        try:
            float(source)
        except ValueError:
            return False

        # 检查目标数据是否符合科学计数法格式
        if not re.match(scientific_notation_pattern, target):
            return False

        # 验证数值是否相等（允许浮点精度误差）
        try:
            source_val = float(source)
            target_val = float(target)
            if abs(source_val - target_val) > 1e-10:  # 允许小的浮点误差
                return False
        except ValueError:
            return False

    return True


def is_semicolon_extraction_conversion(examples):
    """
    判断是否为分号分隔信息提取转换 - 增强版本
    规则：源数据包含分号分隔的多个信息，目标数据是提取或清理其中的某一部分
    """
    string_operations_count = 0
    total_examples = len(examples)

    for exp in examples:
        source = str(exp[0])
        target = str(exp[1])

        # 情况1：分号分隔提取
        if ';' in source:
            # 检查目标数据是否是源数据的子字符串
            if target in source:
                parts = [part.strip() for part in source.split(';')]
                # 检查目标是否匹配某个部分或部分内容的清理版本
                for part in parts:
                    clean_part = re.sub(r'[^a-zA-Z0-9\s\-]', '', part).strip()
                    clean_target = re.sub(r'[^a-zA-Z0-9\s\-]', '', target).strip()
                    if clean_target in clean_part or clean_part in clean_target:
                        string_operations_count += 1
                        break
            continue

        # 情况2：字符替换和清理
        if any(char in source for char in '–-;') and any(char in target for char in '--'):
            # 检查是否主要是字符替换操作
            source_clean = re.sub(r'[^a-zA-Z0-9\s]', '', source).strip()
            target_clean = re.sub(r'[^a-zA-Z0-9\s]', '', target).strip()
            if source_clean == target_clean or target_clean in source_clean:
                string_operations_count += 1
            continue

        # 情况3：文本提取（从复杂字符串中提取简单文本）
        if len(source) > len(target) * 1.5:  # 源明显比目标长
            # 检查目标是否是源的子集
            source_words = set(re.findall(r'[a-zA-Z]+', source.lower()))
            target_words = set(re.findall(r'[a-zA-Z]+', target.lower()))
            if target_words.issubset(source_words) and len(target_words) > 0:
                string_operations_count += 1
            continue

    # 如果大部分示例都可以通过字符串操作完成，则分类为String
    return string_operations_count >= total_examples * 0.8  # 80%的示例符合String特征


def is_all_string_operations(examples):
    """
    检查是否所有转换都可以通过字符串操作完成
    """
    for exp in examples:
        source = str(exp[0])
        target = str(exp[1])

        # 跳过纯数字到纯数字的转换（可能是Numbers类）
        if (source.replace('.', '').replace('-', '').strip().isdigit() and
                target.replace('.', '').replace('-', '').strip().isdigit()):
            return False

        # 检查是否可以通过常见的字符串操作完成
        if not can_be_string_operation(source, target):
            return False

    return True


def can_be_string_operation(source, target):
    """
    检查单个转换是否可以通过字符串操作完成
    """
    source = str(source)
    target = str(target)

    # 情况1：目标完全包含在源中
    if target in source:
        return True

    # 情况2：分号分隔提取
    if ';' in source:
        parts = [part.strip() for part in source.split(';')]
        for part in parts:
            if target.strip() in part or part in target.strip():
                return True

    # 情况3：字符清理和替换
    source_clean = re.sub(r'[^\w\s]', '', source).lower().strip()
    target_clean = re.sub(r'[^\w\s]', '', target).lower().strip()
    if source_clean == target_clean or target_clean in source_clean:
        return True

    # 情况4：单词子集
    source_words = set(re.findall(r'\w+', source.lower()))
    target_words = set(re.findall(r'\w+', target.lower()))
    if target_words.issubset(source_words) and len(target_words) > 0:
        return True

    # 情况5：简单的字符串变换（首字母、缩写等）
    if len(source.split()) > 1 and len(target.split()) == 1:
        # 可能是提取首字母或缩写
        initials = ''.join(word[0] for word in source.split() if word).lower()
        if initials in target.lower():
            return True

    return False


def get_prediction(examples):
    # 首先检查是否为科学计数法转换
    if is_scientific_notation_conversion(examples):
        print("检测到科学计数法转换，直接分类为 Algorithmic")
        return "Algorithmic"

    # 检查是否为分号分隔信息提取转换
    if is_semicolon_extraction_conversion(examples):
        print("检测到分号分隔信息提取转换，直接分类为 String")
        return "String"

    # 新增：检查是否所有转换都可以通过字符串操作完成
    if is_all_string_operations(examples):
        print("检测到所有转换都可以通过字符串操作完成，直接分类为 String")
        return "String"

    cache_file = PROMPT_CACHE_PATH / f"{MODEL_NAME}.pkl"

    # 完全禁用缓存，强制重新调用API
    cache_dict = {}

    mdl = MODEL_NAME

    with open(CODE_BASE_PATH / f"classifier/prompts/{mdl}/class_prompt_{PROMPT_VERSION}.txt", encoding='utf-8') as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

    print("=" * 80)

    # 直接调用API，不检查缓存
    print("调用DeepSeek API...")
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0000001,
        max_tokens=100,
    )
    respond = completion.choices[0].message.content
    print("API调用完成")

    print(f"DeepSeek原始响应: {respond}")

    # 响应解析逻辑
    if "Class:" in respond:
        out = respond.split("Class:")[1].strip()
        out = re.split(r'\s+', out)[0]
    else:
        words = re.split(r'\s+', respond)
        for word in words:
            clean_word = word.strip('.,:;!?()"\'')
            if clean_word in ALLOWED_CLASSES:
                out = clean_word
                break
        else:
            out = re.split(r'\s+', respond.replace("Class:", "").strip())[0]

    print(f"解析后的分类: {out}")
    print("=" * 80)

    if out in ALLOWED_CLASSES:
        return out

    raise ValueError(f"Invalid out: {out}")


all_labels = {}


def save_all_classes():
    for ds_path in DS_PATHS:
        ds_name = pathlib.Path(ds_path).name
        print(f"Working on {ds_name}:")
        cnt = 1
        tables = sample_data(ds_path, EXAMPLE_SIZE, EXAMPLE_SIZE_TYPE)
        for name, table in tables.items():
            gold_label = get_gold_label(name, ds_path)

            # 在调用API前先检查各种已知模式
            if is_scientific_notation_conversion(table['train']):
                print(f"检测到科学计数法转换，直接分类为 Algorithmic")
                prediction = "Algorithmic"
            elif is_semicolon_extraction_conversion(table['train']):
                print(f"检测到分号分隔信息提取转换，直接分类为 String")
                prediction = "String"
            elif is_all_string_operations(table['train']):
                print(f"检测到所有转换都可以通过字符串操作完成，直接分类为 String")
                prediction = "String"
            else:
                prediction = get_prediction(table['train'])

            print(f"{cnt}/{len(tables)}: {name} -> {prediction} (expected {gold_label})")

            cnt += 1

            assert name not in all_labels
            all_labels[name] = {
                "golden_value": gold_label,
                "predicted_value": prediction,
                "full_name": f"{ds_name}/{name}",
            }

    json.dump(all_labels, open(ALL_CLASSES_JSON, 'w'), indent=2)


if __name__ == "__main__":
    save_all_classes()
    report_metrics(ALL_CLASSES_JSON)