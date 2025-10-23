import json
import os
import pathlib
import pickle
import re
import sys

from openai import OpenAI
#准确率87%

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"

# 修复：确保缓存目录存在
PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

with open(BASE_PATH / 'deepseek.key', 'r') as f:
    API_KEY = f.read().strip()

MODEL_NAME = "deepseek-chat"
PROMPT_VERSION = "v002"

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


def get_prediction(examples):
    cache_file = PROMPT_CACHE_PATH / f"{MODEL_NAME}.pkl"

    # 完全禁用缓存，强制重新调用API
    cache_dict = {}

    # 或者如果你想保留缓存功能但确保这次是新的：
    # if os.path.exists(cache_file):
    #     os.remove(cache_file)
    #     print("已删除缓存文件")
    # cache_dict = {}

    mdl = MODEL_NAME

    with open(CODE_BASE_PATH / f"classifier/prompts/{mdl}/class_prompt_{PROMPT_VERSION}.txt", encoding='utf-8') as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

    print("=" * 80)
    # print("提交给DeepSeek的提示词内容：")
    # print(prompt)
    # print("=" * 80)

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