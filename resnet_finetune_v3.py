"""
Исполнитель: Баранов Дмитрий Олегович
Четыре эксперимента:
  1. Feature Extraction      — pretrained, backbone заморожен, только fc обучается
  2. Full Fine-tuning        — pretrained, вся сеть, дифференцированный lr
  3. From Scratch            — без pretrained весов, полное обучение с нуля
  4. Full FT + CIFAR-адаптация — как эксперимент 2, но conv1/maxpool адаптированы
                                 под 32x32
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
print(f"Используем устройство: {DEVICE}")

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


transform_train_224 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

transform_val_224 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Эксперимент 4, вход 32×32
transform_train_32 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

transform_val_32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# 224×224
trainset_224 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train_224)
valset_224   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_val_224)
trainloader_224 = torch.utils.data.DataLoader(trainset_224, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valloader_224   = torch.utils.data.DataLoader(valset_224,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 32×32
trainset_32 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_train_32)
valset_32   = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform_val_32)
trainloader_32 = torch.utils.data.DataLoader(trainset_32, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
valloader_32   = torch.utils.data.DataLoader(valset_32,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Тренировочных данных: {len(trainset_224)}, Валидационных данных: {len(valset_224)}")


# АДАПТАЦИЯ МОДЕЛИ ПОД CIFAR-10
def adapt_for_cifar(model: nn.Module) -> nn.Module:
    """
    Адаптирует ResNet под вход 32×32 (CIFAR-10).

    Проблема: оригинальный ResNet-18 имеет:
      conv1: kernel=7, stride=2  → 224×224 → 112×112
      maxpool: stride=2          → 112×112 → 56×56
    Итого после stem: 56×56. Для 32×32 входа это даст 8×8 после stem —
    слишком агрессивное сжатие, теряется информация

    Решение:
      conv1: kernel=3, stride=1, padding=1  → 32×32 → 32×32 
      maxpool: заменяется Identity
    Residual-блоки (layer1-4) делают даунсемплинг сами
    """
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    model.maxpool = nn.Identity()  # убираем MaxPool — просто пропускаем тензор без изменений
    return model


# ПРОВЕРКА РАЗМЕРНОСТЕЙ
print("\n=== Проверка размерностей ===")

# Стандартная модель
model_check = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_check.fc = nn.Linear(512, NUM_CLASSES)
model_check.eval()
with torch.no_grad():
    out = model_check(torch.randn(1, 3, 224, 224))
print(f"Стандартная модель: вход (1,3,224,224) → выход {tuple(out.shape)}")
assert out.shape == (1, 10)
del model_check
# Адаптивная модель
model_check_adapted = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_check_adapted.fc = nn.Linear(512, NUM_CLASSES)
model_check_adapted = adapt_for_cifar(model_check_adapted)
model_check_adapted.eval()
with torch.no_grad():
    out_adapted = model_check_adapted(torch.randn(1, 3, 32, 32))
print(f"Адаптированная модель: вход (1,3, 32, 32) → выход {tuple(out_adapted.shape)}")
assert out_adapted.shape == (1, 10)
del model_check_adapted


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

    return running_loss / len(dataloader), 100. * correct / total


def eval_epoch(model, criterion, dataloader, device):
    """
    Один проход по валидационной выборке (без обновления весов).

    model.eval() отключает Dropout и переводит BatchNorm в inference-режим
    (использует накопленные running_mean/running_var вместо статистик батча).
    torch.no_grad() отключает вычисление графа градиентов — экономит память и ускоряет.
    """
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


def train_model(model, optimizer, mode_name, trainloader, valloader,
                epochs=EPOCHS, save_path=None):
    """
    Полный цикл обучения.

    Лучшая модель сохраняется по val accuracy, а не по train accuracy.
    Причина: train accuracy может расти за счёт переобучения (overfitting),
    тогда как val accuracy отражает реальное качество обобщения.
    Сохраняя checkpoint при новом максимуме val_acc, мы получаем
    наилучшую модель с точки зрения обобщения.
    """
    criterion = nn.CrossEntropyLoss()
    # CrossEntropyLoss = LogSoftmax + NLLLoss
    # Принимает сырые логиты (не softmax!) и целочисленные метки классов.
    # Внутри считает: loss = -log(exp(logit_true) / sum(exp(logits)))

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, trainloader, DEVICE)
        val_loss, val_acc     = eval_epoch(model, criterion, valloader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        print(f"[{mode_name}] Epoch {epoch+1:>2}/{epochs} | "
              f"Train Loss: {train_loss:.3f}  Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.3f}  Acc: {val_acc:.1f}% | "
              f"{elapsed:.1f}s")

        if save_path is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ Новый рекорд! Сохранено в {save_path} ({val_acc:.1f}%)")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs,
        "best_val_acc": best_val_acc if save_path is not None else max(val_accs),
    }


def get_predictions(model, valloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = model(inputs.to(DEVICE))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_results(results, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)
    epochs_range = range(1, len(results["train_losses"]) + 1)

    ax1.plot(epochs_range, results["train_losses"], label='Train', marker='o', ms=4)
    ax1.plot(epochs_range, results["val_losses"],   label='Val',   marker='o', ms=4)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropyLoss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, results["train_accs"], label='Train', marker='o', ms=4)
    ax2.plot(epochs_range, results["val_accs"],   label='Val',   marker='o', ms=4)
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy, %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe = title.replace(" ", "_").replace("+", "plus")
    plt.savefig(f'plot_{safe}.png', dpi=150)
    plt.show()
    print(f"  График сохранён: plot_{safe}.png")


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
    safe = title.replace(" ", "_").replace("+", "plus")
    plt.savefig(f'confusion_matrix_{safe}.png', dpi=150)
    plt.show()
    print(f"  Confusion matrix сохранена: confusion_matrix_{safe}.png")


def show_errors(model, valloader, title, n=8):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = model(inputs.to(DEVICE))
            _, predicted = outputs.max(1)
            for i in range(len(labels)):
                if predicted[i].cpu() != labels[i] and len(errors) < n:
                    errors.append((inputs[i], labels[i].item(), predicted[i].cpu().item()))
            if len(errors) >= n:
                break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f'Примеры ошибок — {title}')
    mean_, std_ = np.array(CIFAR_MEAN), np.array(CIFAR_STD)

    for idx, (img, true, pred) in enumerate(errors):
        ax = axes[idx // 4][idx % 4]
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = std_ * img_np + mean_   # денормализация
        img_np = np.clip(img_np, 0, 1)
        # Для изображений 224×224 показываем уменьшенную версию (imshow справится)
        ax.imshow(img_np)
        ax.set_title(f'True: {CIFAR10_CLASSES[true]}\nPred: {CIFAR10_CLASSES[pred]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    safe = title.replace(" ", "_").replace("+", "plus")
    plt.savefig(f'errors_{safe}.png', dpi=150)
    plt.show()


def print_model_info(model, name):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемых параметров: {n_train:,} из {n_total:,} ({100*n_train/n_total:.2f}%)")


# ==================== ЭКСПЕРИМЕНТ 1: FEATURE EXTRACTION ====================
print("\n" + "=" * 65)
print("ЭКСПЕРИМЕНТ 1: Feature Extraction")
print("Pretrained backbone заморожен, обучается только fc-слой.")
print("=" * 65)
# Гиперпараметры:
#   lr=1e-3 — стандартный lr для Adam при обучении одного линейного слоя.
#   Большой lr здесь допустим, т.к. backbone не трогаем — нет риска
#   испортить предобученные веса.

model_fe = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
for param in model_fe.parameters():
    param.requires_grad = False          # замораживаем весь backbone
model_fe.fc = nn.Linear(512, NUM_CLASSES)  # новый fc — requires_grad=True по умолчанию
model_fe = model_fe.to(DEVICE)
print_model_info(model_fe, "Feature Extraction")

optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=1e-3)
results_fe = train_model(model_fe, optimizer_fe, "Exp1-FeatExtract",
                         trainloader_224, valloader_224,
                         save_path="best_exp1_feature_extraction.pth")


# ==================== ЭКСПЕРИМЕНТ 2: FULL FINE-TUNING ====================
print("\n" + "=" * 65)
print("ЭКСПЕРИМЕНТ 2: Full Fine-tuning (дифференцированный lr)")
print("Обучается вся сеть. Ранние слои — меньший lr (уже хорошо обучены),")
print("поздние слои и fc — больший lr (нужно переспециализироваться).")
print("=" * 65)
# Гиперпараметры:
#   layer1, layer2: lr=1e-5 — ранние слои выучивают базовые признаки (края, текстуры),
#     они хорошо переносятся, нужны лишь минимальные корректировки.
#   layer3, layer4: lr=1e-4 — более специфичные признаки, чуть активнее подстраиваем.
#   fc: lr=1e-3 — новый слой, обучается с нуля, нужен стандартный lr.

model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_ft.fc = nn.Linear(512, NUM_CLASSES)
model_ft = model_ft.to(DEVICE)
print_model_info(model_ft, "Full Fine-tuning")

optimizer_ft = torch.optim.Adam([
    {'params': model_ft.layer1.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer2.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer3.parameters(), 'lr': 1e-4},
    {'params': model_ft.layer4.parameters(), 'lr': 1e-4},
    {'params': model_ft.fc.parameters(),     'lr': 1e-3},
], lr=1e-4)

results_ft = train_model(model_ft, optimizer_ft, "Exp2-FullFT",
                         trainloader_224, valloader_224,
                         save_path="best_exp2_full_finetuning.pth")


# ==================== ЭКСПЕРИМЕНТ 3: FROM SCRATCH ====================
print("\n" + "=" * 65)
print("ЭКСПЕРИМЕНТ 3: From Scratch (без pretrained весов)")
print("Все веса инициализированы случайно (He initialization).")
print("=" * 65)
# Гиперпараметры:
#   lr=1e-3 — стандартный lr Adam при обучении с нуля.
#   Все слои обучаются одинаково — нет смысла дифференцировать lr,
#   т.к. никакие слои не несут полезных предобученных признаков.

model_scratch = resnet18(weights=None)   # случайная инициализация (He init по умолчанию)
model_scratch.fc = nn.Linear(512, NUM_CLASSES)
model_scratch = model_scratch.to(DEVICE)
print_model_info(model_scratch, "From Scratch")

optimizer_scratch = torch.optim.Adam(model_scratch.parameters(), lr=1e-3)
results_scratch = train_model(model_scratch, optimizer_scratch, "Exp3-FromScratch",
                              trainloader_224, valloader_224,
                              save_path="best_exp3_from_scratch.pth")


# ==================== ЭКСПЕРИМЕНТ 4: FULL FT + CIFAR-АДАПТАЦИЯ ====================
print("\n" + "=" * 65)
print("ЭКСПЕРИМЕНТ 4: Full Fine-tuning + CIFAR-адаптация модели")
print("conv1: 7×7/stride2 → 3×3/stride1, MaxPool убран.")
print("Вход: 32×32 (без Resize). Источник: arXiv:1512.03385, sec. 4.2")
print("=" * 65)

model_adapted = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_adapted.fc = nn.Linear(512, NUM_CLASSES)
model_adapted = adapt_for_cifar(model_adapted)   # <-- ключевое изменение
model_adapted = model_adapted.to(DEVICE)
print_model_info(model_adapted, "Full FT + CIFAR-адаптация")
# Примечание: conv1 заменяется новым слоём (случайная инициализация),
# остальные веса (layer1-4, bn1, fc) остаются pretrained.

optimizer_adapted = torch.optim.Adam([
    {'params': model_adapted.conv1.parameters(),  'lr': 1e-3},   # новый слой
    {'params': model_adapted.layer1.parameters(), 'lr': 1e-5},
    {'params': model_adapted.layer2.parameters(), 'lr': 1e-5},
    {'params': model_adapted.layer3.parameters(), 'lr': 1e-4},
    {'params': model_adapted.layer4.parameters(), 'lr': 1e-4},
    {'params': model_adapted.fc.parameters(),     'lr': 1e-3},
], lr=1e-4)

results_adapted = train_model(model_adapted, optimizer_adapted, "Exp4-Adapted",
                              trainloader_32, valloader_32,
                              save_path="best_exp4_adapted.pth")


# ==================== ИТОГОВАЯ ТАБЛИЦА ====================
print("\n" + "=" * 65)
print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
print("=" * 65)

all_results = {
    "1. Feature Extraction":    results_fe,
    "2. Full Fine-tuning":      results_ft,
    "3. From Scratch":          results_scratch,
    "4. FT + CIFAR-адаптация":  results_adapted,
}

header = f"{'Эксперимент':<26} {'Best Val Acc':>12} {'Final Train Acc':>16} {'Final Val Acc':>14}"
print(header)
print("-" * len(header))
for name, res in all_results.items():
    print(f"{name:<26} {res['best_val_acc']:>11.1f}% "
          f"{res['train_accs'][-1]:>15.1f}% "
          f"{res['val_accs'][-1]:>13.1f}%")

# Сохраняем в JSON
summary = {
    name: {
        "best_val_acc":      res["best_val_acc"],
        "final_train_acc":   res["train_accs"][-1],
        "final_val_acc":     res["val_accs"][-1],
        "final_train_loss":  res["train_losses"][-1],
        "final_val_loss":    res["val_losses"][-1],
    }
    for name, res in all_results.items()
}
with open("experiments_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nСводка сохранена: experiments_summary.json")


# ==================== ГРАФИКИ И ВИЗУАЛИЗАЦИЯ ====================
print("\nСтроим графики для всех экспериментов...")

for title, res in all_results.items():
    plot_results(res, title)

# Confusion matrix и ошибки — для лучшего эксперимента и всех остальных
experiments_vis = [
    (model_fe,      valloader_224, "1. Feature Extraction"),
    (model_ft,      valloader_224, "2. Full Fine-tuning"),
    (model_scratch, valloader_224, "3. From Scratch"),
    (model_adapted, valloader_32,  "4. FT + CIFAR-адаптация"),
]

for model, vloader, title in experiments_vis:
    preds, labels = get_predictions(model, vloader)
    plot_confusion_matrix(preds, labels, title)
    show_errors(model, vloader, title)

print("\n" + "=" * 65)
print("ГОТОВО. Все артефакты сохранены в текущей папке:")
print("  - plot_*.png             — графики loss/accuracy")
print("  - confusion_matrix_*.png — матрицы ошибок")
print("  - errors_*.png           — примеры ошибочных предсказаний")
print("  - best_exp*.pth          — лучшие веса каждого эксперимента")
print("  - experiments_summary.json — сводная таблица метрик")
print("=" * 65)

for name, res in all_results.items():
    print(f"{name:<30} Best Val Acc: {res['best_val_acc']:.1f}%")
