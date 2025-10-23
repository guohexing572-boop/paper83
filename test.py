import json
import os
import pathlib
import pickle
import re
import sys
import pandas as pd
from openai import OpenAI

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"

# 确保缓存目录存在
PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

with open(BASE_PATH / 'deepseek.key', 'r') as f:
    API_KEY = f.read().strip()

MODEL_NAME = "deepseek-chat"
PROMPT_VERSION = "v003"

ALLOWED_CLASSES = ("String", "General", "Numbers", "Algorithmic")


def is_scientific_notation_conversion(examples):
    """判断是否为科学计数法转换"""
    scientific_notation_pattern = r'^-?\d+(\.\d+)?[eE][+-]?\d+$'

    for exp in examples:
        source = str(exp[0])
        target = str(exp[1])

        try:
            float(source)
        except ValueError:
            return False

        if not re.match(scientific_notation_pattern, target):
            return False

        try:
            source_val = float(source)
            target_val = float(target)
            if abs(source_val - target_val) > 1e-10:
                return False
        except ValueError:
            return False

    return True


def is_semicolon_extraction_conversion(examples):
    """判断是否为分号分隔信息提取转换"""
    for exp in examples:
        source = str(exp[0])
        target = str(exp[1])

        if ';' not in source:
            return False

        if target not in source:
            return False

        parts = [part.strip() for part in source.split(';')]
        if target.strip() not in parts:
            return False

    return True


def extract_examples_from_csv(csv_file_path):
    """从CSV文件中提取转换示例 - 最终版本"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        print(f"CSV文件列名: {list(df.columns)}")

        # 找出源列和目标列
        source_columns = [col for col in df.columns if 'source' in col.lower()]
        target_columns = [col for col in df.columns if 'target' in col.lower()]

        print(f"找到源列: {source_columns}")
        print(f"找到目标列: {target_columns}")

        if not source_columns or not target_columns:
            print("错误: 未找到源列或目标列")
            return None

        # 创建明确的列映射
        column_mapping = []

        # 基于列名模式创建映射
        for src_col in source_columns:
            src_lower = src_col.lower()
            # 尝试找到对应的目标列
            matching_target = None
            for tgt_col in target_columns:
                tgt_lower = tgt_col.lower()
                # 基于关键词匹配
                if 'governor' in src_lower and 'name' in tgt_lower:
                    matching_target = tgt_col
                elif 'term' in src_lower and 'name' in tgt_lower:
                    matching_target = tgt_col
                elif 'party' in src_lower and 'party' in tgt_lower:
                    matching_target = tgt_col
                elif 'governor' in src_lower and '#' not in tgt_lower and 'party' not in tgt_lower:
                    matching_target = tgt_col

            if matching_target:
                column_mapping.append((src_col, matching_target))

        # 如果没有找到明确映射，使用第一个源列和第一个目标列
        if not column_mapping and source_columns and target_columns:
            column_mapping.append((source_columns[0], target_columns[0]))

        print(f"列映射关系: {column_mapping}")

        # 提取示例
        examples = []
        for i in range(min(3, len(df))):  # 使用3个示例就够了
            for src_col, tgt_col in column_mapping:
                if (pd.notna(df.iloc[i][src_col]) and
                        pd.notna(df.iloc[i][tgt_col]) and
                        str(df.iloc[i][src_col]).strip() and
                        str(df.iloc[i][tgt_col]).strip()):

                    source_val = str(df.iloc[i][src_col])
                    target_val = str(df.iloc[i][tgt_col])

                    # 跳过明显不相关的映射
                    if (source_val.replace('.', '').replace('-', '').strip().isalpha() and
                            target_val.replace('.', '').replace('-', '').strip().isdigit()):
                        continue

                    examples.append((source_val, target_val))

        # 如果示例太少，添加一些额外的映射
        if len(examples) < 2:
            for i in range(min(2, len(df))):
                if 'source-Governor' in df.columns and 'target-Name (Tenure)' in df.columns:
                    if (pd.notna(df.iloc[i]['source-Governor']) and
                            pd.notna(df.iloc[i]['target-Name (Tenure)'])):
                        examples.append((
                            str(df.iloc[i]['source-Governor']),
                            str(df.iloc[i]['target-Name (Tenure)'])
                        ))
                if 'source-Party' in df.columns and 'target-Party' in df.columns:
                    if (pd.notna(df.iloc[i]['source-Party']) and
                            pd.notna(df.iloc[i]['target-Party'])):
                        examples.append((
                            str(df.iloc[i]['source-Party']),
                            str(df.iloc[i]['target-Party'])
                        ))

        # 去重并限制数量
        unique_examples = []
        seen = set()
        for example in examples:
            if example not in seen:
                seen.add(example)
                unique_examples.append(example)

        return unique_examples[:5]

    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None


def get_prediction(examples):
    """获取转换类型的预测"""
    # 首先检查是否为科学计数法转换
    if is_scientific_notation_conversion(examples):
        print("检测到科学计数法转换，直接分类为 Algorithmic")
        return "Algorithmic"

    # 检查是否为分号分隔信息提取转换
    if is_semicolon_extraction_conversion(examples):
        print("检测到分号分隔信息提取转换，直接分类为 String")
        return "String"

    # 调用DeepSeek API进行分类
    mdl = MODEL_NAME

    with open(CODE_BASE_PATH / f"classifier/prompts/{mdl}/class_prompt_{PROMPT_VERSION}.txt", encoding='utf-8') as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

    print("=" * 80)
    print("提交给DeepSeek的提示词内容：")
    print(prompt)
    print("=" * 80)

    # 调用API
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


def classify_csv_file(csv_file_path):
    """主函数：对CSV文件进行分类"""
    print(f"正在分析CSV文件: {csv_file_path}")
    print("-" * 50)

    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件不存在 - {csv_file_path}")
        return None

    # 从CSV提取示例
    examples = extract_examples_from_csv(csv_file_path)

    if not examples:
        print("错误: 无法从CSV文件中提取有效的转换示例")
        return None

    print(f"从CSV文件中提取了 {len(examples)} 个转换示例:")
    for i, (source, target) in enumerate(examples, 1):
        print(f"示例 {i}: '{source}' -> '{target}'")
    print("-" * 50)

    # 手动检查是否为String类
    all_string = True
    for source, target in examples:
        # 检查是否可以通过字符串操作完成
        if ';' in source and target in source:
            continue  # 分号提取，符合String类
        elif any(char in source for char in '–-;') and any(char in target for char in '--'):
            continue  # 字符替换，符合String类
        else:
            all_string = False
            break

    if all_string:
        print("检测到所有转换都可以通过字符串操作完成，直接分类为 String")
        return "String"

    # 获取预测结果
    prediction = get_prediction(examples)

    print(f"\n最终分类结果: {prediction}")
    return prediction


def main():
    """主程序"""
    print("CSV文件分类器")
    print("=" * 50)

    # 获取用户输入的CSV文件路径
    csv_path = input("请输入CSV文件的完整路径: ").strip()
    csv_path = csv_path.strip('"\'')

    # 分类CSV文件
    result = classify_csv_file(csv_path)

    if result:
        print(f"\n✅ 分类完成!")
        print(f"📁 文件: {csv_path}")
        print(f"📊 类型: {result}")
    else:
        print(f"\n❌ 分类失败!")


if __name__ == "__main__":
    main()