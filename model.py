import torch
import torchvision


class Shape_Effi_B7(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_Effi_B7, self).__init__()
        self.effi_b7 = torchvision.models.efficientnet_b7(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.effi_b7(x)
        x = self.fc(x)

        return x


class Shape_Effi_B0(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_Effi_B0, self).__init__()
        self.effi_b0 = torchvision.models.efficientnet_b0(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.effi_b0(x)
        x = self.fc(x)

        return x


class Shape_ResNet152(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_ResNet152, self).__init__()
        self.resnet152 = torchvision.models.resnet152(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.resnet152(x)
        x = self.fc(x)

        return x


class Shape_ResNet18(torch.nn.Module):
    def __init__(self, class_num):
        super(Shape_ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, class_num)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)

        return x


class MyModel(torch.nn.Module):
    def __init__(self, class_num):
        super(MyModel, self).__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.35),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.35),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.35),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.35),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(p=0.35),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=1),
        )

        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(1, 64)
        self.fc3 = torch.nn.Linear(1, 64)
        self.fc4 = torch.nn.Linear(1, 64)
        self.fc5 = torch.nn.Linear(1, 64)

        self.fc = torch.nn.Linear(1, class_num)

    def forward(self, x):
        x = self.block1(x)
        x1 = x.view(x.size(0), -1)
        print(x1.size())
        x1 = self.fc1(x1)

        x = self.block2(x)
        x2 = x.view(x.size(0), -1)
        print(x2.size())
        x2 = self.fc2(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = x.view(x.size(0), -1)
        print(x.size())
        x = torch.cat([x2, x], dim=1)
        x = self.fc(x)

        return x

