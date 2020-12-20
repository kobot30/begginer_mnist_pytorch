import cv2
import numpy as np
import torch
from model_select import ModelSelectFlag


def test(model, test_loader, optimizer, criterion, device="cpu", flag=None):
    model.eval()
    total_loss = 0
    tp = 0
    total_label = 0
    with torch.no_grad():
        for batch_index, (x_data, y_data) in enumerate(test_loader):
            x, y = x_data.to(device), y_data.to(device)
            optimizer.zero_grad()
            pred_y = model(x)

            if flag == ModelSelectFlag.REGRESSION:
                pred_y = pred_y[:, 0]  # shape = [batch, 1] -> [batch]
                loss = criterion(pred_y, y.float())  # cast float
            elif flag == ModelSelectFlag.CLASSIFICATION:
                loss = criterion(pred_y, y)
            elif flag == ModelSelectFlag.AUTOENCODER:
                loss = criterion(x, pred_y)  # 予測データと入力データを比較
            else:
                raise RuntimeError("loss empty")

            total_loss += loss.item()

            # 分類タスクのみaccuracyを計算
            if flag == ModelSelectFlag.CLASSIFICATION:
                predict_label = pred_y.argmax(1).to('cpu')
                total_label += y_data.shape[0]
                tp += (predict_label == y_data).sum().item()

    # 分類タスクのみaccuracyを出力
    if flag == ModelSelectFlag.CLASSIFICATION:
        acc = tp / total_label
        print('test accuracy = %.3f' % acc)

    return total_loss / batch_index


def test_show(model, model_path, test_loader, device="cpu", flag=None):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for _, (x_data, y_data) in enumerate(test_loader):
            x, y = x_data.to(device), y_data.to(device)

            # 読み込んだ正解ラベルを出力
            print("label_ans = {}".format(y_data.item()))
            pred_y = model(x)

            if flag == ModelSelectFlag.AUTOENCODER:

                # tensor変換時の正規化処理を戻す
                def to_img(x):
                    x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
                    x = x.clamp(0, 1)
                    x = x.view(x.size(0), 1, 28, 28)
                    return x

                # Decoderで復元した出力結果
                pred_y = to_img(pred_y)
                pred = pred_y.to("cpu").detach().numpy()
                pred = pred[0, 0, :, :]
                pred *= 255.0
                pred = pred.astype(np.uint8)

                # 入力画像をtensor型からnumpyへ
                input_image = to_img(x)
                input_image = input_image.to('cpu').detach().numpy()
                input_image = input_image[0, 0, :, :]
                input_image *= 255.0
                input_image = input_image.astype(np.uint8)

                # 入力画像と出力画像の差分
                diff = input_image - pred
                diff_abs = np.abs(diff)

                # 画像左から[入力画像、復元画像、差分画像]
                result = np.zeros((28, 128), dtype=np.uint8)
                result += 255
                result[:28, :28] = input_image
                result[:28, 50:78] = pred
                result[:28, 100:128] = diff_abs

                # write image
                # name = "data/" + str(_) + ".png"
                # result *= 255
                # cv2.imwrite(name, result)

                criterion = torch.nn.MSELoss()
                loss = criterion(x, pred_y)
                print("loss = {}".format(loss.item()))

                print("press enter key. if exit, press [q] key. \n")
                k = cv2.waitKey(-1)
                if k == ord('q'):
                    break

            elif flag == ModelSelectFlag.REGRESSION:
                # 回帰タスクの結果出力
                print("predict (value, label_result) = ({:.3f}, {})".format(pred_y.item(), round(pred_y.item())))
                key_wait()

            elif flag == ModelSelectFlag.CLASSIFICATION:
                # 分類タスクの結果出力
                # print(pred_y)
                predict = pred_y.to("cpu").detach().numpy()[0]
                predict_percentage = predict / sum(predict) * 100.0
                print("predict answer = {}".format((np.argmax(predict_percentage))))
                for i in range(len(predict_percentage)):
                    print("{} = {:.2f} [%]".format(i, predict_percentage[i]))
                key_wait()


def key_wait():
    key = input("press enter key. if exit, press [q] ahd [enter] key. \n")
    if key == "q":
        print("exit")
        exit()
