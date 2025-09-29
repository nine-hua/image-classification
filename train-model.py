import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


def train_model_fixed():
    """修复过拟合问题的训练代码"""

    # 1. 更强的数据增强
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # 随机擦除
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 重新加载数据
    train_dataset = datasets.ImageFolder('dataset/train', transform=transform_train)
    val_dataset = datasets.ImageFolder('dataset/val', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批量
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    # 3. 使用更小的模型或添加正则化
    model = models.resnet34(pretrained=True)  # 改用ResNet34
    # 或者继续用ResNet50但添加更多正则化
    # model = models.resnet50(pretrained=True)

    # 添加Dropout和正则化
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(train_dataset.classes))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. 降低学习率，添加权重衰减
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # 降低学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 5. 添加早停机制
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
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

    early_stopping = EarlyStopping(patience=7)

    # 6. 训练过程
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(50):  # 增加最大轮数，让早停决定
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"早停触发！在第 {epoch + 1} 轮停止训练")
            break

        scheduler.step()
        print("-" * 50)

    return model, train_losses, val_losses, val_accuracies

# 运行修复版本
# model, train_losses, val_losses, val_accuracies = train_model_fixed()
