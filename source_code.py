# Imports

import numpy as np
import torch                    # main torch library
import torch.nn as nn           # neural network modules 
import torch.optim as optim     # optimizers like Adam
import torchvision              # datasets + utilities
import torchvision.transforms as transforms # preprocessing tools

# Load  CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor()       # convert imgae (0-255) -> tensor (0-1)
])

trainset = torchvision.datasets.CIFAR10(
    root = "./data",            # where dataset is stored
    train = True,               # training split
    download = True,            # download if not present
    transform = transform       # apply preprocessing
)

trainloader = torch.utils.data.DataLoader(
    trainset,                   # dataset  object
    batch_size=64,              # number of images per batch
    shuffle = True              # shuffle for better training
)

testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,              # test split
    download = True,
    transform = transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle = False             # no need to shuffle test data
)

# CNN Block

class CNNBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()      # initiate parent class
        
        self.conv = nn.Conv2d(
            in_c, out_c, 3, padding = 1 # 3x3 conv, keep size same
        )
        
        self.bn = nn.BatchNorm2d(out_c) # normalize activations
        self.relu = nn.ReLU()   # non linearity
        
    def forward(self, x):
        x = self.conv(x)        # apply convoluion
        x = self.bn(x)          # stabilize distribution
        x = self.relu(x)        # introduce non linearity
        
        return x
    
# Residual Block

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(c, c, 3, padding=1) # first conv
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x            # save input for skip
        
        out = self.conv1(x)     # first transformation
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)   # second transformation
        out = self.bn2(out)
        
        out += identity         # add original input
        out = self.relu(out)    # activate final result
        
        return out
    
# SE Block
        
class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        
        self.fc1 = nn.Linear(c, c//r)       # reduce channels
        self.fc2 = nn.Linear(c//r, c)       # Expand back
        
    def forward(self, x):
        b, c, h, w = x.shape                # batch, channels, height, width
        
        y = x.view(b, c, -1).mean(dim=2)    # global avg pool -> (b, c)
        
        y = torch.relu(self.fc1(y))         # learn channel importance
        y = torch.sigmoid(self.fc2(y))      # scale between 0 and 1
        
        y = y.view(b, c, 1, 1)              # reshape for broadcasting
        
        return x * y                        # scale each channel
    
class MobileNetBlock(nn.Module):
    def __init__(self, in_c, expand_c, out_c):
        super().__init__()
        
        self.expand = nn.Conv2d(in_c, expand_c, 1)
        self.dw = nn.Conv2d(
            expand_c, expand_c, 3,
            padding=1,
            groups=expand_c
        )
        
        self.project = nn.Conv2d(expand_c, out_c, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.expand(x)
        out = self.relu(out)
        
        out = self.dw(out)
        out = self.relu(out)
        
        out = self.project(out)
        
        if out.shape == identity.shape:
            out += identity
            
        return out
    
# Base CNN Model
class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            CNNBlock(3, 32),
            nn.MaxPool2d(2),
            
            CNNBlock(32, 64),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(64, 10)
        
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Residual Model
class ResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            CNNBlock(3, 32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            
            CNNBlock(32, 64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    
class SECNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            CNNBlock(3, 32),
            SEBlock(32),
            nn.MaxPool2d(2),
            
            CNNBlock(32, 64),
            SEBlock(64),
            nn.MaxPool2d(1)
        )
        
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class MobileNetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            MobileNetBlock(3, 32, 32),
            nn.MaxPool2d(2),

            MobileNetBlock(32, 64, 64),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
            
# Training Function
def train(model, epochs = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(trainloader)    
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss: .4f}")
        

models = {
    "Base": BaseCNN(),
    "Residual": ResCNN(),
    "SE": SECNN(),
    "MobileNet": MobileNetCNN()
}
        
for name, model in models.items():
    print(f"\nTraining {name} model")
    train(model)


def evaluate(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # turn off training behavior

    correct = 0
    total = 0

    with torch.no_grad():  # no gradients needed
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

results = {}

for name, model in models.items():
    print(f"\nTraining {name}")

    train(model)

    acc = evaluate(model, testloader)

    print(f"{name} Accuracy: {acc:.2f}%")

    results[name] = acc
    
print("\nFinal Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")
    
results = {}

for name, model in models.items():
    print(f"\nTraining {name}")

    train(model)

    acc = evaluate(model, testloader)

    results[name] = {
        "accuracy": acc,
    }

# Print nicely
print("\nFinal Comparison:")
for name, r in results.items():
    print(f"{name}: Acc={r['accuracy']:.2f}%")