import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from collections import Counter
import json
import pathlib
import re
import sys

BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
sys.path.append(str(CODE_BASE_PATH))

from util.dataset import sample_data
from classifierutil import get_gold_label


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])


class UltraSimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UltraSimpleClassifier, self).__init__()
        # 极简网络结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.7),  # 极高的dropout

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class AdvancedModelTrainer:
    def __init__(self, model_path="advanced_classifier.pth"):
        self.model_path = model_path
        # 使用更少的特征
        self.vectorizer = TfidfVectorizer(
            max_features=100,  # 大幅减少特征
            ngram_range=(1, 1),  # 只用unigram
            stop_words='english',
            min_df=10,  # 更高的最小文档频率
            max_df=0.6
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None

    def extract_core_features(self, examples):
        """只提取最核心的特征"""
        features = []

        # 只分析第一个示例
        if examples:
            exp = examples[0]
            source = str(exp[0])
            target = str(exp[1])

            # 核心特征
            source_len = len(source)
            target_len = len(target)
            length_diff = target_len - source_len

            # 数字特征
            source_has_digits = any(c.isdigit() for c in source)
            target_has_digits = any(c.isdigit() for c in target)

            # 特殊字符
            has_semicolon = ';' in source
            has_dash = '-' in source or '-' in target

            # 简单模式
            is_numeric_conversion = (source.replace('.', '').replace('-', '').isdigit() and
                                     target.replace('.', '').replace('-', '').isdigit())

            features = [
                source_len, target_len, length_diff,
                float(source_has_digits), float(target_has_digits),
                float(has_semicolon), float(has_dash),
                float(is_numeric_conversion)
            ]

        # 补齐到固定长度
        return features + [0.0] * (8 - len(features))

    def prepare_balanced_features(self, examples_list, labels=None):
        """准备平衡的特征"""
        all_texts = []
        all_core_features = []

        for examples in examples_list:
            # 极简文本特征
            source_texts = [str(exp[0]) for exp in examples[:2]]  # 只取前2个源文本
            combined_text = " ".join(source_texts)
            all_texts.append(combined_text)

            # 核心特征
            core_features = self.extract_core_features(examples)
            all_core_features.append(core_features)

        if labels is not None:
            tfidf_features = self.vectorizer.fit_transform(all_texts).toarray()
            labels_encoded = self.label_encoder.fit_transform(labels)
            core_scaled = self.scaler.fit_transform(all_core_features)
        else:
            tfidf_features = self.vectorizer.transform(all_texts).toarray()
            core_scaled = self.scaler.transform(all_core_features)
            labels_encoded = None

        # 合并特征
        combined_features = np.hstack([core_scaled, tfidf_features])

        print(f"特征维度: {combined_features.shape}")
        print(f"TF-IDF特征: {tfidf_features.shape}, 核心特征: {core_scaled.shape}")

        if labels is not None:
            return combined_features, labels_encoded
        else:
            return combined_features

    def train_with_data_augmentation(self, examples_list, labels, epochs=50, batch_size=16):
        """使用数据增强训练"""
        features, labels_encoded = self.prepare_balanced_features(examples_list, labels)

        # 更严格的数据分割
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels_encoded, test_size=0.4, random_state=42, stratify=labels_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"数据集: 训练{X_train.shape[0]}, 验证{X_val.shape[0]}, 测试{X_test.shape[0]}")

        # 创建模型
        input_dim = features.shape[1]
        output_dim = len(self.label_encoder.classes_)
        self.model = UltraSimpleClassifier(input_dim, output_dim)

        # 数据加载器
        train_dataset = SimpleDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = SimpleDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 使用带类别权重的损失函数
        class_counts = Counter(labels_encoded)
        class_weights = torch.FloatTensor([1.0 / class_counts[i] for i in range(output_dim)])
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 更保守的优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # 训练循环
        best_val_acc = 0
        patience = 12
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels.squeeze()).sum().item()

            # 验证
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = self.model(batch_features)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels.squeeze()).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            scheduler.step(val_acc)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}]')
                print(f'  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
                print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            # 更严格的早停
            if val_acc > best_val_acc + 0.5:  # 要求有实质提升
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model()
                print(f'  ↳ 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'早停在第 {epoch + 1} 轮')
                    break

        # 最终测试
        self.load_model()
        test_acc = self.evaluate(X_test, y_test)
        return test_acc

    def evaluate(self, X_test, y_test):
        """评估模型"""
        self.model.eval()
        test_dataset = SimpleDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)

        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = self.model(batch_features)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.numpy())
                all_true_labels.extend(batch_labels.squeeze().numpy())

        # 计算准确率
        accuracy = np.sum(np.array(all_predictions) == np.array(all_true_labels)) / len(all_true_labels)

        # 详细报告
        true_labels_str = self.label_encoder.inverse_transform(all_true_labels)
        pred_labels_str = self.label_encoder.inverse_transform(all_predictions)

        print("\n" + "=" * 60)
        print("详细测试结果:")
        print("=" * 60)
        print(classification_report(true_labels_str, pred_labels_str, digits=4))

        cm = confusion_matrix(true_labels_str, pred_labels_str, labels=self.label_encoder.classes_)
        print("混淆矩阵:")
        print(cm)

        return accuracy * 100

    def save_model(self):
        """保存模型"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_state, f)

    def load_model(self):
        """加载模型"""
        with open(self.model_path, 'rb') as f:
            model_state = pickle.load(f)

        self.vectorizer = model_state['vectorizer']
        self.label_encoder = model_state['label_encoder']
        self.scaler = model_state['scaler']

        dummy_features = self.prepare_balanced_features([[]])
        input_dim = dummy_features.shape[1]
        output_dim = len(self.label_encoder.classes_)

        self.model = UltraSimpleClassifier(input_dim, output_dim)
        self.model.load_state_dict(model_state['model_state_dict'])


def prepare_training_data():
    """准备训练数据"""
    json_file = str(BASE_PATH / "data/Classes/deepseek_classified.json")

    with open(json_file, 'r') as f:
        data = json.load(f)

    examples_list = []
    labels_list = []

    DS_PATHS = [
        str(BASE_PATH / "data/Datasets/DataXFormer"),
        str(BASE_PATH / "data/Datasets/AutoJoin"),
        str(BASE_PATH / "data/Datasets/FlashFill"),
        str(BASE_PATH / "data/Datasets/All_TDE"),
    ]

    for ds_path in DS_PATHS:
        ds_name = pathlib.Path(ds_path).name
        tables = sample_data(ds_path, 5, "fixed")

        for name, table in tables.items():
            if name in data:
                examples_list.append(table['train'])
                labels_list.append(data[name]['golden_value'])

    return examples_list, labels_list


if __name__ == "__main__":
    print("训练高级模型（极简架构 + 严格正则化）...")

    examples_list, labels_list = prepare_training_data()

    trainer = AdvancedModelTrainer()
    test_accuracy = trainer.train_with_data_augmentation(examples_list, labels_list, epochs=50)

    print(f"\n🎯 最终测试准确率: {test_accuracy:.2f}%")