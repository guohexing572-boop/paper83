import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

# 添加必要的路径
BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.parent.parent.absolute()
CODE_BASE_PATH = pathlib.Path(__file__).absolute().parent.parent.absolute()
sys.path.append(str(CODE_BASE_PATH))

from util.dataset import sample_data
from classifierutil import get_gold_label


class BalancedExampleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])


class LightweightClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(LightweightClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class OptimizedExampleClassifierTrainer:
    def __init__(self, model_path="optimized_example_classifier.pth"):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # 固定特征数量
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.feature_dim = None

    def calculate_class_weights(self, labels):
        """类别权重计算"""
        class_counts = Counter(labels)
        total_samples = len(labels)

        weights = {}
        for class_label, count in class_counts.items():
            weights[class_label] = np.sqrt(total_samples / (2.0 * count))

        class_weights = [weights[cls] for cls in sorted(class_counts.keys())]
        print(f"Class weights: {dict(zip(sorted(class_counts.keys()), class_weights))}")
        return torch.FloatTensor(class_weights)

    def extract_handcrafted_features(self, examples):
        """提取手工特征 - 固定维度"""
        # 固定特征数量：每个示例13个特征，最多5个示例 = 65个特征
        fixed_features = [0.0] * 65  # 初始化为0

        for i, exp in enumerate(examples):
            if i >= 5:  # 只取前5个示例
                break

            source = str(exp[0])
            target = str(exp[1])

            # 基础特征
            source_len = len(source)
            target_len = len(target)
            length_ratio = target_len / (source_len + 1e-8)

            # 字符类型特征
            source_digit_ratio = sum(c.isdigit() for c in source) / (len(source) + 1e-8)
            target_digit_ratio = sum(c.isdigit() for c in target) / (len(target) + 1e-8)
            source_alpha_ratio = sum(c.isalpha() for c in source) / (len(source) + 1e-8)
            target_alpha_ratio = sum(c.isalpha() for c in target) / (len(target) + 1e-8)

            # 特殊字符特征
            special_chars = ';,-_.:()[]{}'
            source_special_count = sum(source.count(c) for c in special_chars)
            target_special_count = sum(target.count(c) for c in special_chars)

            # 模式特征
            is_numeric_source = float(source.replace('.', '').replace('-', '').isdigit())
            is_numeric_target = float(target.replace('.', '').replace('-', '').isdigit())
            has_semicolon = float(';' in source)
            has_dash = float('-' in source or '-' in target)

            # 填充到固定位置
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

    def prepare_all_features(self, all_examples_list, labels=None):
        """一次性准备所有特征以确保维度一致"""
        all_texts = []
        all_handcrafted = []

        # 第一步：收集所有文本和手工特征
        for examples in all_examples_list:
            # 文本特征
            combined_text = " ".join([f"{exp[0]} {exp[1]}" for exp in examples])
            all_texts.append(combined_text)

            # 手工特征
            handcrafted_features = self.extract_handcrafted_features(examples)
            all_handcrafted.append(handcrafted_features)

        # 第二步：TF-IDF特征（在所有数据上拟合）
        if labels is not None:
            tfidf_features = self.vectorizer.fit_transform(all_texts).toarray()
            labels_encoded = self.label_encoder.fit_transform(labels)
            handcrafted_scaled = self.scaler.fit_transform(all_handcrafted)
        else:
            tfidf_features = self.vectorizer.transform(all_texts).toarray()
            handcrafted_scaled = self.scaler.transform(all_handcrafted)
            labels_encoded = None

        # 第三步：合并特征
        combined_features = np.hstack([handcrafted_scaled, tfidf_features])
        self.feature_dim = combined_features.shape[1]

        print(f"总特征维度: {combined_features.shape}")
        print(f"手工特征: {handcrafted_scaled.shape}, TF-IDF特征: {tfidf_features.shape}")

        if labels is not None:
            return combined_features, labels_encoded
        else:
            return combined_features

    def prepare_features_consistent(self, examples_list, labels=None):
        """为单个数据集准备特征"""
        all_texts = []
        all_handcrafted = []

        for examples in examples_list:
            combined_text = " ".join([f"{exp[0]} {exp[1]}" for exp in examples])
            all_texts.append(combined_text)

            handcrafted_features = self.extract_handcrafted_features(examples)
            all_handcrafted.append(handcrafted_features)

        # 使用预先训练好的vectorizer和scaler
        tfidf_features = self.vectorizer.transform(all_texts).toarray()
        handcrafted_scaled = self.scaler.transform(all_handcrafted)

        combined_features = np.hstack([handcrafted_scaled, tfidf_features])

        if labels is not None:
            labels_encoded = self.label_encoder.transform(labels)
            return combined_features, labels_encoded
        else:
            return combined_features

    def train_with_validation(self, train_examples, train_labels, val_examples, val_labels,
                              epochs=50, batch_size=16, learning_rate=0.001):
        """使用验证集的训练过程"""
        # 首先在所有训练数据上拟合特征处理器
        print("正在准备特征...")
        all_examples = train_examples + val_examples
        all_labels = train_labels + val_labels

        # 在所有数据上拟合特征处理器
        all_features, all_labels_encoded = self.prepare_all_features(all_examples, all_labels)

        # 分割特征
        train_size = len(train_examples)
        X_train = all_features[:train_size]
        y_train = all_labels_encoded[:train_size]
        X_val = all_features[train_size:]
        y_val = all_labels_encoded[train_size:]

        print(f"训练特征维度: {X_train.shape}")
        print(f"验证特征维度: {X_val.shape}")

        # 计算类别权重
        class_weights = self.calculate_class_weights(train_labels)

        # 创建带权重的采样器
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # 创建模型
        input_dim = X_train.shape[1]
        output_dim = len(self.label_encoder.classes_)
        self.model = LightweightClassifier(input_dim, output_dim, dropout_rate=0.4)

        print(f"模型输入维度: {input_dim}, 输出维度: {output_dim}")

        # 创建数据加载器
        train_dataset = BalancedExampleDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        val_dataset = BalancedExampleDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # 训练循环
        self.model.train()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()

                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels.squeeze())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels.squeeze()).sum().item()

            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels.squeeze())
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels.squeeze()).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            scheduler.step(avg_val_loss)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}]')
                print(f'  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')

            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model()
                print(f'  ↳ 保存最佳模型 (Val Loss: {best_val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'早停在第 {epoch + 1} 轮')
                    break

        self.is_trained = True
        print("训练完成!")

        # 加载最佳模型
        self.load_model()

    def predict(self, examples_list):
        """预测分类"""
        if not self.is_trained:
            raise ValueError("模型未训练")

        features = self.prepare_features_consistent(examples_list)

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            outputs = self.model(features_tensor)
            _, predicted = torch.max(outputs, 1)

        predicted_labels = self.label_encoder.inverse_transform(predicted.numpy())
        return predicted_labels

    def evaluate(self, examples_list, true_labels):
        """评估模型性能"""
        predictions = self.predict(examples_list)

        print("\n" + "=" * 60)
        print("最终模型评估结果:")
        print("=" * 60)
        print(classification_report(true_labels, predictions, digits=4))

        cm = confusion_matrix(true_labels, predictions, labels=self.label_encoder.classes_)
        print("混淆矩阵:")
        print(cm)

        accuracy = np.sum(np.array(predictions) == np.array(true_labels)) / len(true_labels)
        print(f"总体准确率: {accuracy:.4f}")

        return predictions, accuracy

    def save_model(self):
        """保存模型"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_dim': self.feature_dim
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
        self.feature_dim = model_state['feature_dim']

        output_dim = len(self.label_encoder.classes_)
        self.model = LightweightClassifier(self.feature_dim, output_dim)
        self.model.load_state_dict(model_state['model_state_dict'])
        self.is_trained = True


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


def train_final_model():
    """训练最终模型"""
    print("训练优化后的模型...")

    # 准备数据
    examples_list, labels_list = prepare_training_data()

    # 三层分割: 训练/验证/测试
    X_temp, X_test, y_temp, y_test = train_test_split(
        examples_list, labels_list, test_size=0.15, random_state=42, stratify=labels_list
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"类别分布: {Counter(labels_list)}")

    # 训练模型
    trainer = OptimizedExampleClassifierTrainer()
    trainer.train_with_validation(X_train, y_train, X_val, y_val, epochs=50, batch_size=16)

    # 最终测试
    print("\n" + "=" * 60)
    print("在测试集上的最终表现:")
    print("=" * 60)
    predictions, accuracy = trainer.evaluate(X_test, y_test)

    return trainer, accuracy


if __name__ == "__main__":
    final_model, final_accuracy = train_final_model()
    print(f"\n最终测试准确率: {final_accuracy:.4f}")