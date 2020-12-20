import torch
from tqdm import tqdm
from torch import nn, optim
from torchvision.utils import save_image
from torch.autograd import Variable


# GAN学習
def train_gan_run(n_epochs, g_model, d_model, mnist_dim, batch_size, z_dim, train_loader, device):
    # loss (binary cross entropy)
    criterion = nn.BCELoss()

    # optimizer
    lr = 0.0002
    g_optimizer = optim.Adam(g_model.parameters(), lr=lr)
    d_optimizer = optim.Adam(d_model.parameters(), lr=lr)

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            d_losses, g_losses = [], []
            for batch_idx, (x, _) in enumerate(train_loader):
                d_losses.append(d_train(x, g_model, d_model, mnist_dim, batch_size, z_dim, device, criterion, d_optimizer))
                g_losses.append(g_train(g_model, d_model, batch_size, z_dim, device, criterion, g_optimizer))

            d_loss = torch.mean(torch.FloatTensor(d_losses)).item()
            g_loss = torch.mean(torch.FloatTensor(g_losses)).item()
            pbar.set_postfix(d_loss=d_loss, g_loss=g_loss)

            # 画像生成テスト
            with torch.no_grad():
                tmp_size = 32
                test_z = Variable(torch.randn(tmp_size, z_dim).to(device))
                generated = g_model(test_z)
                name = "data/sample_" + str(epoch) + ".png"
                save_image(generated.view(generated.size(0), 1, 28, 28), name)

    return g_losses, d_losses


# 識別機学習
def d_train(x, g_model, d_model, mnist_dim, batch_size, z_dim, device, criterion, d_optimizer):
    d_model.zero_grad()

    # 本物データ(訓練データ)学習
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    d_output = d_model(x_real)
    d_real_loss = criterion(d_output, y_real)
    d_real_score = d_output

    # 偽物データ(生成器)学習
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake, y_fake = g_model(z), Variable(torch.zeros(batch_size, 1).to(device))

    d_output = d_model(x_fake)
    d_fake_loss = criterion(d_output, y_fake)
    d_fake_score = d_output

    # 識別器のみパラメータ更新
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()


# 生成器学習
def g_train(g_model, d_model, batch_size, z_dim, device, criterion, g_optimizer):
    g_model.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    # 生成器が生成したものに対して識別器の結果ｑをもとに損失計算
    g_output = g_model(z)
    d_output = d_model(g_output)
    g_loss = criterion(d_output, y)

    # 生成器のみパラメータを更新
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()
