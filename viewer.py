import numpy as np
import cv2
import cvui
import torch
from torchvision import transforms

# NN model
from model.classification import Classification
from model.auto_encoder import AutoEncoder
from model.cam_conv_classification import CamConvClassification
from model.cam import Cam
from model_select import ModelSelectFlag

# クリック周辺の塗りつぶす範囲
PAINT_SIZE = 4


def predict(model, img, device="cpu", flag=None):
    model.eval()

    # 手書き文字をMNISTのフォーマットに変換する
    input_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # カラー画像から白黒画像へ
    input_resize = cv2.resize(input_gray, (28, 28))  # 28x28ピクセルへ
    input_resize = np.reshape(input_resize, (28, 28, 1))  # 1chであるように配列を修正

    # 学習時と同じテンソル型に変換する
    to_tensor = transforms.ToTensor()(input_resize)
    tensor_normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))(to_tensor)
    input_resize = tensor_normalize.to("cpu").detach().numpy()
    input_resize = np.reshape(input_resize, (1, 1, 28, 28))  # [batch, ch, height, width]
    tensor = torch.from_numpy(input_resize)
    input_tensor = tensor.to(device)  # 推論に使用する入力画像のTensor型

    with torch.no_grad():
        if flag == ModelSelectFlag.CLASSIFICATION:
            pred_y = model(input_tensor)
            predict = pred_y.to("cpu").detach().numpy()[0]
            predict_percentage = predict / sum(predict) * 100.0
            for i in range(len(predict_percentage)):
                print("{} = {:.2f} [%]".format(i, predict_percentage[i]))
            print("predict answer = {}".format((np.argmax(predict_percentage))))
            return np.argmax(predict_percentage)

        if flag == ModelSelectFlag.AUTOENCODER:
            pred_y = model(input_tensor)

            # 正規化処理を戻す(tensor)
            def convert_normalize_img_tensor(x):
                x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
                x = x.clamp(0, 1)
                x = x.view(x.size(0), 1, 28, 28)
                return x

            # Decoderで復元した出力結果
            pred_y = convert_normalize_img_tensor(pred_y)
            pred = pred_y.to("cpu").detach().numpy()
            pred = pred[0, 0, :, :]
            pred *= 255.0
            pred = pred.astype(np.uint8)
            return pred

        if flag == ModelSelectFlag.CAM:
            torch.set_grad_enabled(True)  # for turn off torch.no_grad()
            cam = Cam(model)
            output = model(input_tensor)
            values, idx = output.max(dim=1)
            print("predict number = {}".format(idx.item()))

            model.zero_grad()
            output[0, idx].backward(retain_graph=True)  # output[batch_index, label]
            out = cam.get_cam(0)

            cam_img = cv2.resize(out.cpu().data.numpy(), (28, 28))

            # 正規化処理を戻す(numpy)
            def convert_normalize_img(img):
                img = 0.5 * (img + 1)  # [-1,1] => [0, 1]
                img *= 255
                img = img.astype(np.uint8)
                return img

            cam_img = convert_normalize_img(cam_img)
            cam_color_map = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
            return cam_color_map


if __name__ == '__main__':
    # 横幅 500px 縦幅 400pxのメインウィンドウを作成
    window_w = 500
    window_h = 400
    frame = np.zeros((window_h, window_w, 3), np.uint8)

    # メインウィンドウの初期化
    WINDOW_NAME = "viewer"
    cvui.init(WINDOW_NAME)
    frame[:] = (255, 255, 255)  # 画面全体を白で一旦初期化
    frame[70:210, 70:210, :] = 0  # 手書き領域の黒初期化
    frame[70:210, 280:420, :] = 0  # 結果表示領域の黒初期化
    cvui.text(frame, 70, 50, "Draw(Input)", 0.5, 0x000000)
    cvui.text(frame, 280, 50, "Predict", 0.5, 0x000000)

    # device config (CPU or GPUを使用するか識別している)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use device = {}'.format(device))

    # 分類モデル読み込み
    classification_model_path = "data/model_classification.pth"
    model_cl = Classification().to(device)
    model_cl.load_state_dict(torch.load(classification_model_path))

    # Auto Encoder モデル読み込み
    auto_encoder_model_path = "data/model_auto_encoder.pth"
    model_ae = AutoEncoder().to(device)
    model_ae.load_state_dict(torch.load(auto_encoder_model_path))

    # Class Activation Map モデル読み込み
    cam_model_path = "data/model_cam.pth"
    model_cam = torch.nn.DataParallel(CamConvClassification().to(device))
    model_cam.load_state_dict(torch.load(cam_model_path))

    while True:

        # quit
        k = cv2.waitKey(10)
        if k == 27:  # esc key
            break
        elif k == ord('q'):  # q key
            break

        # click event
        if cvui.mouse(cvui.IS_DOWN):
            point = cvui.mouse()
            x, y = point.x, point.y
            if 70 <= x <= 210 and 70 <= y <= 210:
                frame[y - PAINT_SIZE:y + PAINT_SIZE, x - PAINT_SIZE:x + PAINT_SIZE, :] = 255

        # erase
        if cvui.button(frame, 85, 250, 100, 100, "erase"):
            frame[70:210, 70:210, :] = 0
            print("erase")

        # classification
        if cvui.button(frame, 280, 250, 140, 30, "classify"):
            input_img = frame[70:210, 70:210, :]
            number = predict(model_cl, input_img, device, ModelSelectFlag.CLASSIFICATION)
            frame[70:210, 280:420, :] = 0
            cvui.text(frame, 310, 100, str(number), 4, 0xffffff)

        # auto encoder
        if cvui.button(frame, 280, 300, 140, 30, "auto encoder"):
            input_img = frame[70:210, 70:210, :]
            pred_img = predict(model_ae, input_img, device, ModelSelectFlag.AUTOENCODER)
            pred_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
            pred_resize = cv2.resize(pred_color, (140, 140))
            frame[70:210, 280:420, :] = pred_resize

        # class activation map
        if cvui.button(frame, 280, 350, 140, 30, "class activation map"):
            input_img = frame[70:210, 70:210, :]
            pred_img = predict(model_cam, input_img, device, ModelSelectFlag.CAM)
            pred_resize = cv2.resize(pred_img, (140, 140))
            add_image = cv2.addWeighted(input_img, 0.5, pred_resize, 0.5, 2)
            frame[70:210, 280:420, :] = add_image

        # 画面更新
        cvui.imshow(WINDOW_NAME, frame)
