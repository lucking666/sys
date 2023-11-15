import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class dA(nn.Module):
    def __init__(self, in_features, out_features):
        super(dA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_features, in_features),
            nn.ReLU()
        )

        self.decoder[0].weight.data = self.encoder[0].weight.data.transpose(0, 1)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)


class SdA(nn.Module):
    def __init__(self, config):
        super(SdA, self).__init__()

        layers = []
        in_features = config.input_features
        for out_features in config.hidden_features:
            layer = dA(in_features, out_features)
            in_features = out_features
            layers.append(layer)
        layers.append(nn.Linear(in_features, config.classes))
        self.layers = nn.Sequential(*layers)

        if config.is_train:
            self.mse_criterion = nn.MSELoss()
            self.ce_criterion = nn.CrossEntropyLoss()

            self.da_optimizers = []
            for layer in self.layers[:-1]:
                optimizer = optim.SGD(layer.parameters(), lr=config.lr,
                                      momentum=config.momentum, weight_decay=config.weight_decay)
                self.da_optimizers.append(optimizer)

            sda_params = []
            for layer in self.layers[:-1]:
                sda_params.extend(layer.encoder.parameters())
            sda_params.extend(self.layers[-1].parameters())
            self.sda_optimizer = optim.SGD(sda_params, lr=config.lr,
                                           momentum=config.momentum, weight_decay=config.weight_decay)

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = layer.encoder(h)
        return self.layers[-1](h)

def denoise_data(X, model):
    # 将数据转化为 PyTorch 的 Variable
    X_var = torch.tensor(X, dtype=torch.float32)

    # 在训练集上进行去噪
    denoised_data = X_var.detach().numpy()
    for layer in model.layers[:-1]:
        denoised_data = layer.encoder(torch.tensor(denoised_data, dtype=torch.float32)).detach().numpy()

    return denoised_data

def train_and_denoise(X_train, Y_train, X_test, config):
    # 数据处理
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 初始化模型
    sda_model = SdA(config)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(sda_model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        outputs = sda_model(X_train_tensor)

        # 计算损失
        loss = criterion(outputs, X_train_tensor)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))

    # 在训练集上进行去噪
    denoised_X_train = denoise_data(X_train, sda_model)

    # 在测试集上进行去噪
    denoised_X_test = denoise_data(X_test, sda_model)

    return denoised_X_train, denoised_X_test

class Config:
    def __init__(self):
        self.input_features = 1
        self.hidden_features = [16, 8]  # 例如，有两个隐藏层，分别为 16 和 8 个特征
        self.classes = 1
        self.is_train = True
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-5


