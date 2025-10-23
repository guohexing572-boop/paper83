import json
import os
import pathlib
import pickle
import re
import sys

from openai import OpenAI
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

#准确率93%

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
PROMPT_CACHE_PATH = BASE_PATH / "cache/classifier_prompts"

# 修复：确保缓存目录存在
PROMPT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

with open(BASE_PATH / 'deepseek.key', 'r') as f:
    API_KEY = f.read().strip()

MODEL_NAME = "deepseek-chat"
PROMPT_VERSION = "v003"

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


# 加载你的训练好的模型
class LightweightClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(LightweightClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class LocalModelPredictor:
    def __init__(self, model_path="optimized_example_classifier.pth"):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.scaler = None
        self.feature_dim = None
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_state = pickle.load(f)

            self.vectorizer = model_state['vectorizer']
            self.label_encoder = model_state['label_encoder']
            self.scaler = model_state['scaler']
            self.feature_dim = model_state['feature_dim']

            output_dim = len(self.label_encoder.classes_)
            self.model = LightweightClassifier(self.feature_dim, output_dim)
            self.model.load_state_dict(model_state['model_state_dict'])
            self.model.eval()

            print(f"本地模型加载成功，特征维度: {self.feature_dim}")
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            self.model = None

    def extract_handcrafted_features(self, examples):
        """提取手工特征 - 固定维度"""
        fixed_features = [0.0] * 65

        for i, exp in enumerate(examples):
            if i >= 5:
                break

            source = str(exp[0])
            target = str(exp[1])

            source_len = len(source)
            target_len = len(target)
            length_ratio = target_len / (source_len + 1e-8)

            source_digit_ratio = sum(c.isdigit() for c in source) / (len(source) + 1e-8)
            target_digit_ratio = sum(c.isdigit() for c in target) / (len(target) + 1e-8)
            source_alpha_ratio = sum(c.isalpha() for c in source) / (len(source) + 1e-8)
            target_alpha_ratio = sum(c.isalpha() for c in target) / (len(target) + 1e-8)

            special_chars = ';,-_.:()[]{}'
            source_special_count = sum(source.count(c) for c in special_chars)
            target_special_count = sum(target.count(c) for c in special_chars)

            is_numeric_source = float(source.replace('.', '').replace('-', '').isdigit())
            is_numeric_target = float(target.replace('.', '').replace('-', '').isdigit())
            has_semicolon = float(';' in source)
            has_dash = float('-' in source or '-' in target)

            start_idx = i * 13
            fixed_features[start_idx:start_idx + 13] = [
                source_len, target_len, length_ratio,
                source_digit_ratio, target_digit_ratio,
                source_alpha_ratio, target_alpha_ratio,
                source_special_count, target_special_count,
                is_numeric_source, is_numeric_target,
                has_semicolon, has_dash
            ]

        return fixed_features

    def predict_with_confidence(self, examples):
        """使用本地模型预测并返回预测结果和置信度"""
        if self.model is None:
            return None, 0.0

        try:
            all_texts = []
            all_handcrafted = []

            combined_text = " ".join([f"{exp[0]} {exp[1]}" for exp in examples])
            all_texts.append(combined_text)

            handcrafted_features = self.extract_handcrafted_features(examples)
            all_handcrafted.append(handcrafted_features)

            tfidf_features = self.vectorizer.transform(all_texts).toarray()
            handcrafted_scaled = self.scaler.transform(all_handcrafted)

            combined_features = np.hstack([handcrafted_scaled, tfidf_features])

            with torch.no_grad():
                features_tensor = torch.FloatTensor(combined_features)
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probabilities, 1)

            predicted_label = self.label_encoder.inverse_transform(predicted.numpy())[0]
            confidence = max_prob.item()

            return predicted_label, confidence

        except Exception as e:
            print(f"本地模型预测失败: {e}")
            return None, 0.0


# 初始化本地模型预测器
local_predictor = LocalModelPredictor()


def call_deepseek_api(examples):
    """调用DeepSeek API进行预测"""
    # 完全禁用缓存
    print("禁用缓存，强制调用API...")

    mdl = MODEL_NAME

    with open(CODE_BASE_PATH / f"classifier/prompts/{mdl}/class_prompt_{PROMPT_VERSION}.txt", encoding='utf-8') as f:
        pmpt = f.read()

    str_examp = ""

    for exp in examples:
        str_examp += f"(\"{exp[0]}\" -> \"{exp[1]}\"),"

    prompt = pmpt.format(examples=str_examp)

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

    if out in ALLOWED_CLASSES:
        return out

    raise ValueError(f"Invalid out: {out}")


def get_prediction(examples):
    # 第一步：使用本地小模型预测
    local_prediction, confidence = local_predictor.predict_with_confidence(examples)

    # 如果本地模型置信度大于90%，直接返回结果
    if confidence > 0.9:
        print(f"✅ 本地模型高置信度预测: {local_prediction} (置信度: {confidence:.4f})")
        print("=" * 80)
        final_prediction = local_prediction
        print(f"DEBUG: get_prediction 返回: {final_prediction}")
        return final_prediction

    # 第二步：如果置信度≤90%，调用大模型API
    print(f"⚠️ 本地模型置信度较低 ({confidence:.4f})，调用大模型API...")
    if local_prediction:
        print(f"   本地模型建议: {local_prediction}")

    api_prediction = call_deepseek_api(examples)
    print("=" * 80)
    print(f"DEBUG: get_prediction 返回: {api_prediction}")
    return api_prediction


all_labels = {}


def save_all_classes():
    for ds_path in DS_PATHS:
        ds_name = pathlib.Path(ds_path).name
        print(f"Working on {ds_name}:")
        cnt = 1
        tables = sample_data(ds_path, EXAMPLE_SIZE, EXAMPLE_SIZE_TYPE)
        for name, table in tables.items():
            gold_label = get_gold_label(name, ds_path)

            # 直接使用新的预测流程：小模型 → 大模型
            print(f"DEBUG: 开始处理表 {name}")
            prediction = get_prediction(table['train'])
            print(f"DEBUG: save_all_classes 收到预测结果: {prediction}")

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