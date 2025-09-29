import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class ImageClassifierTrainer:
    def __init__(self, data_dir, model_name='resnet50', num_epochs=50, batch_size=32, learning_rate=0.001):
        """
        图像分类器训练类

        Args:
            data_dir: 数据目录，应包含train和val子目录
            model_name: 模型名称 ('resnet18', 'resnet34', 'resnet50', 'efficientnet_b0')
            num_epochs: 最大训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 初始化变量
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = None
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }

    def setup_data_loaders(self):
        """设置数据加载器"""
        print("设置数据加载器...")

        # 训练集数据增强
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)  # 随机擦除
        ])

        # 验证集预处理
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 创建数据集
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=val_transform
        )

        # 获取类别名称
        self.class_names = train_dataset.classes
        print(f"发现 {len(self.class_names)} 个类别: {self.class_names}")

        # 检查数据集大小
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")

        # 检查类别分布
        self.print_class_distribution(train_dataset, "训练集")
        self.print_class_distribution(val_dataset, "验证集")

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return len(train_dataset), len(val_dataset)

    def print_class_distribution(self, dataset, name):
        """打印类别分布"""
        class_counts = {}
        for _, label in dataset:
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"\n{name}类别分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} 张")

    def create_model(self):
        """创建模型"""
        print(f"创建模型: {self.model_name}")

        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            num_features = self.model.fc.in_features
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
        elif self.model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        # 替换分类层，添加正则化
        if 'efficientnet' in self.model_name:
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.class_names))
            )
        else:
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.class_names))
            )

        self.model = self.model.to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")

    def setup_training(self):
        """设置训练组件"""
        print("设置训练组件...")

        # 损失函数 - 可以根据类别不平衡情况使用加权损失
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=7,
            verbose=True,
            min_lr=1e-7
        )

        # 早停机制
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 打印进度
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 收集预测结果用于详细分析
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total

        return avg_val_loss, val_accuracy, all_preds, all_labels

    def train(self):
        """完整训练流程"""
        print("开始训练...")
        print("=" * 60)

        # 设置数据和模型
        train_size, val_size = self.setup_data_loaders()
        self.create_model()
        self.setup_training()

        # 训练循环
        best_val_acc = 0.0
        best_model_state = None
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            # 训练一个epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # 验证
            val_loss, val_acc, all_preds, all_labels = self.validate()

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"🎉 新的最佳验证准确率: {best_val_acc:.2f}%")

            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} 完成 ({epoch_time:.1f}s)")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.2e}")

            # 每5个epoch打印详细报告
            if (epoch + 1) % 5 == 0:
                print("\n" + "=" * 50)
                print("详细分类报告:")
                print(classification_report(all_labels, all_preds,
                                            target_names=self.class_names,
                                            zero_division=0))
                print("=" * 50)

            # 早停检查
            if self.early_stopping(val_loss):
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break

            print("-" * 60)

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ 已加载最佳模型 (验证准确率: {best_val_acc:.2f}%)")

        total_time = time.time() - start_time
        print(f"\n🎯 训练完成！总耗时: {total_time / 60:.1f} 分钟")
        print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")

        return best_val_acc

    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_history['train_loss']) + 1)

        # 损失曲线
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(epochs, self.train_history['val_accuracy'], 'g-', label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('验证准确率')
        ax2.legend()
        ax2.grid(True)

        # 学习率曲线
        ax3.plot(epochs, self.train_history['learning_rates'], 'm-', label='学习率')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('学习率变化')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)

        # 损失平滑曲线
        if len(epochs) > 5:
            window_size = min(5, len(epochs) // 4)
            train_loss_smooth = self.smooth_curve(self.train_history['train_loss'], window_size)
            val_loss_smooth = self.smooth_curve(self.train_history['val_loss'], window_size)

            ax4.plot(epochs, train_loss_smooth, 'b-', label='训练损失(平滑)')
            ax4.plot(epochs, val_loss_smooth, 'r-', label='验证损失(平滑)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss (Smoothed)')
            ax4.set_title('平滑损失曲线')
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("训练历史图表已保存为 'training_history.png'")

    def smooth_curve(self, points, factor=0.9):
        """平滑曲线"""
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def evaluate_model(self):
        """详细评估模型"""
        print("正在进行详细模型评估...")

        val_loss, val_acc, all_preds, all_labels = self.validate()

        print(f"\n📊 最终评估结果:")
        print(f"验证准确率: {val_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}")

        # 分类报告
        print(f"\n📋 详细分类报告:")
        print(classification_report(all_labels, all_preds,
                                    target_names=self.class_names,
                                    zero_division=0))

        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("混淆矩阵已保存为 'confusion_matrix.png'")

        return val_acc, val_loss

    def save_model(self, save_dir='saved_models'):
        """保存模型和相关信息"""
        print(f"保存模型到 {save_dir}/...")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 1. 保存PyTorch模型
        model_path = os.path.join(save_dir, 'model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'train_history': self.train_history
        }, model_path)
        print(f"✅ PyTorch模型已保存: {model_path}")

        # 2. 保存ONNX模型
        self.export_to_onnx(save_dir)

        # 3. 保存模型信息
        model_info = {
            'model_name': self.model_name,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_shape': [1, 3, 224, 224],
            'input_size': 224,
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'training_params': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': len(self.train_history['train_loss']),
                'best_val_accuracy': max(self.train_history['val_accuracy'])
            }
        }

        info_path = os.path.join(save_dir, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"✅ 模型信息已保存: {info_path}")

        # 4. 保存类别名称
        class_names_path = os.path.join(save_dir, 'class_names.txt')
        with open(class_names_path, 'w', encoding='utf-8') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        print(f"✅ 类别名称已保存: {class_names_path}")

        # 5. 保存训练历史
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"✅ 训练历史已保存: {history_path}")

        print(f"\n🎉 模型保存完成！保存位置: {save_dir}/")

        return save_dir

    def export_to_onnx(self, save_dir):
        """导出ONNX模型"""
        try:
            import torch.onnx

            # 设置模型为评估模式
            self.model.eval()

            # 创建示例输入
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

            # 导出ONNX
            onnx_path = os.path.join(save_dir, 'model.onnx')
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"✅ ONNX模型已保存: {onnx_path}")

            # 验证ONNX模型
            self.verify_onnx_model(onnx_path, dummy_input)

        except ImportError:
            print("❌ ONNX导出失败: 请安装onnx库 (pip install onnx)")
        except Exception as e:
            print(f"❌ ONNX导出失败: {str(e)}")

    def verify_onnx_model(self, onnx_path, dummy_input):
        """验证ONNX模型"""
        try:
            import onnx
            import onnxruntime as ort

            # 检查ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # 创建ONNX Runtime会话
            ort_session = ort.InferenceSession(onnx_path)

            # 比较PyTorch和ONNX输出
            with torch.no_grad():
                pytorch_output = self.model(dummy_input).cpu().numpy()

            onnx_input = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]

            # 检查输出差异
            diff = np.abs(pytorch_output - onnx_output).max()
            if diff < 1e-5:
                print(f"✅ ONNX模型验证成功 (最大差异: {diff:.2e})")
            else:
                print(f"⚠️  ONNX模型输出差异较大: {diff:.2e}")

        except ImportError:
            print("⚠️  无法验证ONNX模型: 请安装onnx和onnxruntime")
        except Exception as e:
            print(f"⚠️  ONNX模型验证失败: {str(e)}")


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def load_and_test_model(model_path, test_image_path, class_names_path=None):
    """加载模型并测试单张图片"""
    print("加载模型进行测试...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint['model_name']
    class_names = checkpoint['class_names']

    # 重建模型
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(class_names))
        )
    # 添加其他模型的重建逻辑...

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载和预测图片
    from PIL import Image
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()

    print(f"预测结果: {predicted_class}")
    print(f"置信度: {confidence_score:.4f}")

    # 显示前3个预测
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    print("\n前3个预测:")
    for i in range(3):
        class_name = class_names[top3_indices[0][i].item()]
        prob = top3_prob[0][i].item()
        print(f"  {i + 1}. {class_name}: {prob:.4f}")

    return predicted_class, confidence_score
