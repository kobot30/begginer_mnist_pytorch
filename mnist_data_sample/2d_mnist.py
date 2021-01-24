import cv2
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# MNIST
dataset_train = MNIST('data/', train=True, download=True, transform=transform)

# EMNIST
# dataset_train = EMNIST('data/', train=True, download=True, transform=transform, split="byclass")

# KMNIST
# dataset_train = KMNIST('data/', train=True, download=True, transform=transform)

# Fashion-MNIST
# dataset_train = FashionMNIST('data/', train=True, download=True, transform=transform)

# QMNIST
# dataset_train = QMNIST('data/', train=True, download=True, transform=transform)


train_loader = DataLoader(dataset_train, batch_size=10, shuffle=False)
# test_loader = DataLoader(dataset_test)

# 訓練用データを保持するtrain_loaderからバッチサイズ分の画像とラベルを取り出す
# batch_size=50なら50個の画像(images)とラベル(labels)のセットを取り出している
data_iter = iter(train_loader)
images, labels = data_iter.next()

# 画像をindexの順番で表示する
for index, (image, label) in enumerate(zip(images, labels)):
    # tensor型の画像をnumpyとして取り扱う
    npimg = image.to('cpu').detach().numpy()

    # (Channel, Height, Width -> Height, Width, Channel) tensor型からnumpy型に幅、高さ、チャンネルの順番を入れ替える
    npimg = npimg.transpose((1, 2, 0))
    # npimg = npimg.transpose((2, 1, 0))  # EMNIST H,Wの順番を交換する

    # tensor型で0〜1の範囲で輝度値が表現されているため0〜255の範囲に戻す
    npimg *= 255

    # 画像を表示する(28x28ピクセル) & 何かキーを押すまで待機
    print("index = {}, label = {}".format(index, label))
    cv2.imshow("mnist image", npimg)
    save_name = "data/2d_mnist/" + str(index) + "_" + str(label.item()) + ".jpg"
    cv2.imwrite(save_name, npimg)
    k = cv2.waitKey(-1)
    if k == ord('q'):  # q を押したときは終了
        exit(1)

    # save_name = "../data/2d_mnist/" + str(index) + "_" + str(label.item()) + ".jpg"
    # cv2.imwrite(save_name, npimg)
