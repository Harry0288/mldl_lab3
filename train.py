from models.model import CustomNet
import torch
import wandb
from torch import nn
import torch.nn.functional as F

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...
        # compute prediction and loss
        outputs = model(inputs) # prediction
        # criterion is the loss function
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)


model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

wandb.init("lab3")
config = wandb.config
config.learning_rate = 0.001

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)
    print(epoch)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)
    wandb.log({"val_accuracy": val_accuracy, "best_acc": best_acc})


print(f'Best validation accuracy: {best_acc:.2f}%')
