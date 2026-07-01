"""
Fine-tuning ResNet-18 на CIFAR-10
Сравнение трёх режимов обучения:
  1. Feature Extraction  — только fc-слой
  2. Full Fine-tuning    — вся сеть, дифференцированный learning rate
  3. From Scratch        — без pretrained весов, обучение с нуля

"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
import time

EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем: {DEVICE}")

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Статистика CIFAR-10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Размер тренировочной выборки: {len(trainset)}\n, Размер валидационной выборки: {len(valset)}")


# Проверка размерности изображения

_dummy_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
_dummy_model.fc = nn.Linear(512, NUM_CLASSES)
_dummy_model.eval()

dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR размер
print("Вход:", dummy_input.shape)

dummy_resized = nn.functional.interpolate(dummy_input, size=224, mode='bilinear', align_corners=False)
print("Вход после Resize:", dummy_resized.shape)

with torch.no_grad():
    dummy_output = _dummy_model(dummy_resized)
print("Выход модели:", dummy_output.shape)
assert dummy_output.shape == (1, NUM_CLASSES), "Размерность выхода не совпадает с ожидаемой!"
print("Проверка пройдена: (1,3,32,32) --[Resize+Model]--> (1,10)\n")
del _dummy_model


def train_epoch(model, optimizer, criterion, dataloader, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def eval_epoch(model, criterion, dataloader, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, optimizer, mode_name, epochs=EPOCHS, save_path=None):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, optimizer, criterion, trainloader, DEVICE)
        val_loss, val_acc = eval_epoch(model, criterion, valloader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        print(f"[{mode_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.3f} Acc: {val_acc:.1f}% | "
              f"{elapsed:.1f}s")

        if save_path is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Новый рекорд! Модель сохранена в {save_path} ({val_acc:.1f}%)")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc if save_path is not None else max(val_accs),
    }


def get_predictions(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_results(results, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)
    epochs_range = range(1, len(results["train_losses"]) + 1)

    ax1.plot(epochs_range, results["train_losses"], label='Train')
    ax1.plot(epochs_range, results["val_losses"], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs_range, results["train_accs"], label='Train')
    ax2.plot(epochs_range, results["val_accs"], label='Val')
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    safe_title = title.replace(" ", "_")
    plt.savefig(f'{safe_title}.png')
    plt.show()


def plot_confusion_matrix(preds, labels, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES)
    plt.title(f'Confusion Matrix — {title}')
    plt.ylabel('Настоящий класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    safe_title = title.replace(" ", "_")
    plt.savefig(f'confusion_matrix_{safe_title}.png')
    plt.show()


def show_errors(model, title, n=8):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs_dev = inputs.to(DEVICE)
            outputs = model(inputs_dev)
            _, predicted = outputs.max(1)
            for i in range(len(labels)):
                if predicted[i].cpu() != labels[i] and len(errors) < n:
                    errors.append((inputs[i], labels[i].item(), predicted[i].cpu().item()))
            if len(errors) >= n:
                break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Ошибки модели — {title}')
    mean = np.array(CIFAR_MEAN)
    std = np.array(CIFAR_STD)

    for idx, (img, true, pred) in enumerate(errors):
        ax = axes[idx // 4][idx % 4]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
        ax.set_title(f'True: {CIFAR10_CLASSES[true]}\nPred: {CIFAR10_CLASSES[pred]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    safe_title = title.replace(" ", "_")
    plt.savefig(f'errors_{safe_title}.png')
    plt.show()


# РЕЖИМ 1: FEATURE EXTRACTION 
print("\n" + "=" * 60)
print("РЕЖИМ 1: Feature Extraction")
print("=" * 60)

model_fe = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model_fe.parameters():
    param.requires_grad = False

model_fe.fc = nn.Linear(512, NUM_CLASSES)  # requires_grad=True
model_fe = model_fe.to(DEVICE)

optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=0.001)
results_fe = train_model(model_fe, optimizer_fe, "Feature Extraction",
                          save_path="best_model_feature_extraction.pth")

# РЕЖИМ 2: FULL FINE-TUNING
print("\n" + "=" * 60)
print("РЕЖИМ 2: Full Fine-tuning")
print("=" * 60)

model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_ft.fc = nn.Linear(512, NUM_CLASSES)
model_ft = model_ft.to(DEVICE)

optimizer_ft = torch.optim.Adam([
    {'params': model_ft.layer1.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer2.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer3.parameters(), 'lr': 1e-4},
    {'params': model_ft.layer4.parameters(), 'lr': 1e-4},
    {'params': model_ft.fc.parameters(),     'lr': 1e-3},
], lr=1e-4)

results_ft = train_model(model_ft, optimizer_ft, "Full Fine-tuning",
                          save_path="best_model_full_finetuning.pth")

# РЕЖИМ 3: FROM SCRATCH (без pretrained) 
print("\n" + "=" * 60)
print("РЕЖИМ 3: From Scratch (без pretrained весов)")
print("=" * 60)

model_scratch = resnet18(weights=None)  # случайная инициализация всех весов
model_scratch.fc = nn.Linear(512, NUM_CLASSES)
model_scratch = model_scratch.to(DEVICE)

optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=1e-3)
results_scratch = train_model(model_scratch, optimizer_scratch, "From Scratch",
                               save_path="best_model_from_scratch.pth")

# ИТОГОВАЯ ТАБЛИЦА 
print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
print("=" * 60)

all_results = {
    "Feature Extraction": results_fe,
    "Full Fine-tuning": results_ft,
    "From Scratch": results_scratch,
}

print(f"{'Эксперимент':<22} {'Best Val Acc':<14} {'Final Train Acc':<18} {'Final Val Acc':<14}")
print("-" * 70)
for name, res in all_results.items():
    print(f"{name:<22} {res['best_val_acc']:<14.1f} {res['train_accs'][-1]:<18.1f} {res['val_accs'][-1]:<14.1f}")

# Сохраняем таблицу и метрики в JSON для отчёта/презентации
summary = {
    name: {
        "best_val_acc": res["best_val_acc"],
        "final_train_acc": res["train_accs"][-1],
        "final_val_acc": res["val_accs"][-1],
        "final_train_loss": res["train_losses"][-1],
        "final_val_loss": res["val_losses"][-1],
    }
    for name, res in all_results.items()
}
with open("experiments_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nСводка сохранена в experiments_summary.json")

# ВИЗУАЛИЗАЦИЯ 
print("\nСтроим графики и confusion matrix для всех трёх экспериментов...")

plot_results(results_fe, "Feature Extraction")
plot_results(results_ft, "Full Fine-tuning")
plot_results(results_scratch, "From Scratch")

preds_fe, labels_fe = get_predictions(model_fe)
preds_ft, labels_ft = get_predictions(model_ft)
preds_scratch, labels_scratch = get_predictions(model_scratch)

plot_confusion_matrix(preds_fe, labels_fe, "Feature Extraction")
plot_confusion_matrix(preds_ft, labels_ft, "Full Fine-tuning")
plot_confusion_matrix(preds_scratch, labels_scratch, "From Scratch")

show_errors(model_fe, "Feature Extraction")
show_errors(model_ft, "Full Fine-tuning")
show_errors(model_scratch, "From Scratch")

print("\n" + "=" * 60)
print("Графики, confusion matrix и лучшие модели сохранены!")
print("=" * 60)

print(f"Итог Feature Extraction — Best Val Acc: {results_fe['best_val_acc']:.1f}%")
print(f"Итог Full Fine-tuning   — Best Val Acc: {results_ft['best_val_acc']:.1f}%")
print(f"Итог From Scratch       — Best Val Acc: {results_scratch['best_val_acc']:.1f}%")
