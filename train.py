import torchvision.models as models
import torch
from torchvision import datasets
from torchvision.models import MobileNet_V3_Small_Weights
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 定义设备
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 使用预训练的mobilenet_v3权重，并修改最后一个输出层
model = models.mobilenet.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1, )
model.num_classes = 5
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 5)
model.to(device)
#
Train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=8),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

Test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#
train_dataset = datasets.ImageFolder('./dataset/train', transform=Train_transform)
test_dataset = datasets.ImageFolder("./dataset/test", transform=Test_transform)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 5e-5)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 20

print("==============================")
print(" start train!")
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print(" Train Done!")
print("==============================")
model.to("cpu")
input_tensor = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, input_tensor)

traced_script_module.save("MobileNetV3.pt")

print("mode has save to MobileNetV3.pt")