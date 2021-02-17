import click
import matplotlib.pyplot as plt
import cv2
import numpy as np

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

# NN model
from model.classification import Classification
from model.regression import Regression
from model.auto_encoder import AutoEncoder
from model.conv_auto_encoder import ConvAutoEncoder
from model.gan import Generator, Discriminator
from model.cam import Cam
from model.cam_conv_classification import CamConvClassification
from model.unet import Unet

# utils
from model_select import ModelSelectFlag
from train import train_run
from train_gan import train_gan_run
from train_cam import train_cam_run
from test import test_show
from mnist_dataset import MnistDataset, get_image_mask_set
from train_seg import train_seg_run

# lossグラフの描画
def loss_graph(train_loss, test_loss, save_name, label_1='train loss', label_2='test loss'):
    fig = plt.figure()
    plt.plot(range(len(train_loss)), train_loss, marker='.', label=label_1)
    plt.plot(range(len(test_loss)), test_loss, marker='x', label=label_2)
    plt.title('Train and Test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(save_name)


@click.command()
@click.option('--test_flag', '-t', is_flag=False)
@click.option('--convolution', '-conv', is_flag=False)
@click.option('--classification', '-c', is_flag=False)
@click.option('--regression', '-r', is_flag=False)
@click.option('--auto_encoder', '-a', is_flag=False)
@click.option('--gan', '-g', is_flag=False)
@click.option('--cam', '-m', is_flag=False)
@click.option('--segmentation', '-seg', is_flag=False)
def main(test_flag, convolution, classification, regression, auto_encoder, gan, cam, segmentation):

    # device config (CPU or GPUを使用するか識別している)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use device = {}'.format(device))

    # 学習時のデフォルトパラメータ
    n_epochs = 30
    batch_size = 500

    '''
    <データセットの変換処理の内容を示す>
    
    1.「transforms.ToTensor()」によりデータをPyTorchが使用するTensor型に変換する
    2.「transforms.Normalize(~)」によりデータを指定値で正規化する
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    '''
    <MNISTデータセットを読み込む>
    
    root: データを保存するフォルダパスを指定する
    train: 訓練用データ(True) or テスト用データ(False)かを指定する
    download: 指定パスにデータがなければダウンロードするかどうか（基本的にはTrueでOK)
    transform: 事前に指定したデータの変換処理を指定する
    '''
    dataset_train = MNIST(root='data/', train=True, download=True, transform=transform)
    dataset_test = MNIST(root='data/', train=False, download=True, transform=transform)

    '''
    <MNISTデータセットをDataLoaderとしてPyTorchで取り扱う>
    
    batch_size: データ取り出し時のbatch_size
    shuffle: データのindexシャッフルの有効化
    num_workers : CPUコア使用数(高速化に寄与)
    pin_memory : automatic memory pinningの有効化(高速化に寄与)
    '''
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1, pin_memory=True)

    # 分類
    if classification:
        model = Classification().to(device)
        model_path = "data/model_classification.pth"
        if test_flag:
            test_show(model, model_path, test_loader, device, flag=ModelSelectFlag.CLASSIFICATION)
        else:
            train_loss, test_loss = train_run(n_epochs, train_loader, test_loader, model, device, flag=ModelSelectFlag.CLASSIFICATION)
            torch.save(model.state_dict(), model_path)
            graph_save_path = "data/loss_classification.png"
            loss_graph(train_loss, test_loss, graph_save_path)

    # 回帰
    elif regression:
        model = Regression().to(device)
        model_path = "data/model_regression.pth"
        if test_flag:
            test_show(model, model_path, test_loader, device, flag=ModelSelectFlag.REGRESSION)
        else:
            train_loss, test_loss = train_run(n_epochs, train_loader, test_loader, model, device, flag=ModelSelectFlag.REGRESSION)
            torch.save(model.state_dict(), model_path)
            graph_save_path = "data/loss_regression.png"
            loss_graph(train_loss, test_loss, graph_save_path)

    # Auto Encoder
    elif auto_encoder:
        model_path = "data/model_auto_encoder.pth"
        model = AutoEncoder().to(device)
        if convolution:
            model_path = "data/model_conv_auto_encoder.pth"
            model = ConvAutoEncoder().to(device)

        if test_flag:
            test_show(model, model_path, test_loader, device, flag=ModelSelectFlag.AUTOENCODER)
        else:
            train_loss, test_loss = train_run(n_epochs, train_loader, test_loader, model, device, flag=ModelSelectFlag.AUTOENCODER)
            torch.save(model.state_dict(), model_path)
            graph_save_path = "data/loss_auto_encoder.png"
            loss_graph(train_loss, test_loss, graph_save_path)

    # Generative Adversarial Networks
    elif gan:
        n_epochs = 200
        mnist_dim = 28 * 28  # mnist image size = 1*28*28
        z_dim = 100  # 潜在変数
        g_model = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)  # 生成器
        d_model = Discriminator(mnist_dim).to(device)  # 識別器
        g_losses, d_losses = train_gan_run(n_epochs, g_model, d_model, mnist_dim, batch_size, z_dim, train_loader, device)
        graph_save_path = "data/loss_gan.png"
        loss_graph(g_losses, d_losses, graph_save_path, label_1="g_loss", label_2="d_loss")

    # Class Activation Map
    elif cam:
        n_epochs = 10
        model_path = 'data/model_cam.pth'
        model = nn.DataParallel(CamConvClassification().to(device))
        if test_flag:
            model.load_state_dict(torch.load(model_path))
            cam = Cam(model)

            for _, [x_data, y_data] in enumerate(test_loader):
                x, y = x_data.to(device), y_data.to(device)
                output = model.forward(x)

                for j in range(len(y_data)):
                    print(j)
                    model.zero_grad()
                    output[j, y[j]].backward(retain_graph=True)  # 正解ラベルをbackward
                    out = cam.get_cam(j)
                    cam.visualize(out, x[j])

        else:
            train_cam_run(n_epochs, model, train_loader, test_loader, device)
            torch.save(model.state_dict(), model_path)

    # セグメンテーション
    elif segmentation:
        n_epochs = 10
        batch_size = 50
        model_path = 'data/model_seg.pth'
        model = Unet(n_class=2).to(device)

        # MNISTデータセットから一部をndarray型で取得
        mnist_train_images = dataset_train.data.numpy()[:500] # 0-500までの学習用データを使用する
        mnist_test_images = dataset_test.data.numpy()[:batch_size] # 0-batch_sizeまでのテスト用データを使用する

        # RGB画像とマスク画像の2種類に変換する
        train_images, train_masks = get_image_mask_set(mnist_train_images)
        test_images, test_masks = get_image_mask_set(mnist_test_images)           

        # 自前のMnistDatasetを使用しPyTorchのデータローダーとして準備する
        mnist_train = MnistDataset(train_images, train_masks, transform=transform) 
        mnist_test = MnistDataset(test_images, test_masks, transform=transform) 
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, num_workers=1, pin_memory=True)

        if test_flag:
            model.load_state_dict(torch.load(model_path))
            model.eval() 

            color = np.asarray([(0, 0, 0), (0, 0, 255)]) # 背景:黒、検出領域:赤
            for _, [inputs, labels] in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                pred = model(inputs)
                pred = pred.data.cpu().numpy()

                for mask in pred:
                    # mask convert color
                    _, height, width = mask.shape
                    colorimg = np.zeros((height, width, 3), dtype=np.uint8)
                    for y in range(height):
                        for x in range(width):
                           colorimg[y,x,:] = color[np.argmax(mask[:, y, x])]
                    
                    pred_result = cv2.resize(colorimg, (128, 128))
                    cv2.imshow("pred_result", pred_result)
                    k = cv2.waitKey(-1)
                    if k == ord('q'):
                        break
        else:
            model = train_seg_run(n_epochs, model, train_loader, device=device)
            torch.save(model.state_dict(), model_path)
    else:
        raise AssertionError("flag error")


if __name__ == '__main__':
    main()
