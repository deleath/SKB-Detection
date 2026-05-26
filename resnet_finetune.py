import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ==================== НАСТРОЙКИ ====================
EPOCHS = 10
BATCH_SIZE = 64
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем: {DEVICE}")

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ==================== ДАННЫЕ ====================
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_val)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== ФУНКЦИИ ====================
def train_model(model, optimizer, mode_name, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # --- Обучение ---
        model.train()
        running_loss, correct, total = 0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        # --- Валидация ---
        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = running_loss / len(valloader)
        val_acc = 100. * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[{mode_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.3f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.3f} Acc: {val_acc:.1f}%")

    return train_losses, val_losses, train_accs, val_accs

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
    epochs = range(1, len(results[0]) + 1)

    ax1.plot(epochs, results[0], label='Train')
    ax1.plot(epochs, results[1], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, results[2], label='Train')
    ax2.plot(epochs, results[3], label='Val')
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
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
    plt.savefig(f'confusion_matrix_{title.replace(" ", "_")}.png')
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
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for idx, (img, true, pred) in enumerate(errors):
        ax = axes[idx // 4][idx % 4]
        img = img.numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f'True: {CIFAR10_CLASSES[true]}\nPred: {CIFAR10_CLASSES[pred]}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'errors_{title.replace(" ", "_")}.png')
    plt.show()

# ==================== РЕЖИМ 1: FEATURE EXTRACTION ====================
print("\n=== РЕЖИМ 1: Feature Extraction ===")
model_fe = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Замораживаем все слои
for param in model_fe.parameters():
    param.requires_grad = False

# Заменяем последний слой
model_fe.fc = nn.Linear(512, NUM_CLASSES)
model_fe = model_fe.to(DEVICE)

# Обучаем только fc
optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=0.001)
results_fe = train_model(model_fe, optimizer_fe, "Feature Extraction")

# ==================== РЕЖИМ 2: FULL FINE-TUNING ====================
print("\n=== РЕЖИМ 2: Full Fine-tuning ===")
model_ft = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_ft.fc = nn.Linear(512, NUM_CLASSES)
model_ft = model_ft.to(DEVICE)

# Дифференцированный lr: ранние слои учатся медленнее
optimizer_ft = torch.optim.Adam([
    {'params': model_ft.layer1.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer2.parameters(), 'lr': 1e-5},
    {'params': model_ft.layer3.parameters(), 'lr': 1e-4},
    {'params': model_ft.layer4.parameters(), 'lr': 1e-4},
    {'params': model_ft.fc.parameters(),     'lr': 1e-3},
], lr=1e-4)

results_ft = train_model(model_ft, optimizer_ft, "Full Fine-tuning")

# ==================== ВИЗУАЛИЗАЦИЯ ====================
plot_results(results_fe, "Feature Extraction")
plot_results(results_ft, "Full Fine-tuning")

preds_fe, labels_fe = get_predictions(model_fe)
preds_ft, labels_ft = get_predictions(model_ft)

plot_confusion_matrix(preds_fe, labels_fe, "Feature Extraction")
plot_confusion_matrix(preds_ft, labels_ft, "Full Fine-tuning")

show_errors(model_fe, "Feature Extraction")
show_errors(model_ft, "Full Fine-tuning")

print(f"\nИтог Feature Extraction — Val Acc: {results_fe[3][-1]:.1f}%")
print(f"Итог Full Fine-tuning   — Val Acc: {results_ft[3][-1]:.1f}%")