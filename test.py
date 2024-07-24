import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel()
    model = nn.DataParallel(model)
    model.to(device)

    # 创建数据
    inputs = torch.randn(64, 10).to(device)
    labels = torch.randn(64, 1).to(device)

    # 训练模型
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == "__main__":
    main()
