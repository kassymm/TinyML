"""Shared utilities for CIFAR-10 model comparisons."""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm

# ======================== Config ========================
BATCH_SIZE = 128
NUM_EPOCHS = 150
LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2
SEED = 42

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ======================== Data ========================
def get_dataloaders(data_root="../data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    return trainloader, testloader


# ======================== Helpers ========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def model_size_kb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_size   = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buf_size) / 1024

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def print_summary(model, acc, label="Model"):
    n  = count_parameters(model)
    kb = model_size_kb(model)
    print(f"\n{'='*50}")
    print(f"{label} Results")
    print(f"{'='*50}")
    print(f"  Parameters:    {n:,}")
    print(f"  Model Size:    {kb:.1f} KB")
    print(f"  Test Accuracy: {acc:.2f}%")
    assert n  < 125_000, f"CONSTRAINT VIOLATED — too many parameters: {n}"
    assert kb < 500,     f"CONSTRAINT VIOLATED — model too large: {kb:.1f} KB"
    assert acc >= 85.0,  f"CONSTRAINT VIOLATED — accuracy too low: {acc:.2f}%"
    print(f"  All constraints satisfied.")
    print(f"{'='*50}\n")


# ======================== Training ========================
def train(model, trainloader, testloader, device,
          num_epochs=NUM_EPOCHS, lr=LR,
          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
          label_smoothing=LABEL_SMOOTHING,
          save_path="best_model.pth"):

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        batch_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}",
                          leave=False, unit="batch")
        for images, labels in batch_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        avg_loss = running_loss / len(trainloader)
        lr_cur   = scheduler.get_last_lr()[0]
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}",
                               lr=f"{lr_cur:.6f}",
                               best=f"{best_acc:.2f}%")

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            acc = evaluate(model, testloader, device)
            epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                   acc=f"{acc:.2f}%",
                                   best=f"{best_acc:.2f}%")
            tqdm.write(f"Epoch {epoch+1:3d}/{num_epochs} | "
                       f"Loss: {avg_loss:.4f} | "
                       f"Acc: {acc:.2f}% | LR: {lr_cur:.6f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return evaluate(model, testloader, device)


def evaluate_pytorch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total