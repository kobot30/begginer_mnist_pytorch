import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


class Cam():
    def __init__(self, model):
        self.gradient = []
        self.h = model.module.layer[-1].register_backward_hook(self.save_gradient)

    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output = args[2]
        self.gradient.append(grad_output[0])

    def get_gradient(self, idx):
        return self.gradient[idx]

    def remove_hook(self):
        self.h.remove()

    def get_cam(self, idx):
        grad = self.get_gradient(idx)  # [batch, feature, (image width, height)]
        alpha = torch.sum(grad, dim=3, keepdim=True)
        alpha = torch.sum(alpha, dim=2, keepdim=True)  # [batch, feature, 1, 1]

        cam = alpha[idx] * grad[idx]  # 該当するbatch_indexの値を乗算する
        cam = torch.sum(cam, dim=0)
        cam = self.normalize_cam(cam)

        self.remove_hook()
        return cam

    @staticmethod
    def normalize_cam(x):
        x = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        # x[x < torch.max(x)] = -1
        return x

    @staticmethod
    def visualize(cam_img, img_var):
        cam_img = cv2.resize(cam_img.cpu().data.numpy(), (28, 28))
        x = img_var[0, :, :].cpu().data.numpy()

        def convert_normalize_img(img):
            img = 0.5 * (img + 1)  # [-1,1] => [0, 1]
            img *= 255
            img = img.astype(np.uint8)
            return img

        # Cam画像
        cam_img = convert_normalize_img(cam_img)
        cam_color_map = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        cam_color_map = cv2.resize(cam_color_map, (128, 128))

        # 入力画像
        x = convert_normalize_img(x)
        x_resize = cv2.resize(x, (128, 128))
        x_resize = cv2.cvtColor(x_resize, cv2.COLOR_GRAY2BGR)

        # 合成
        add_image = cv2.addWeighted(x_resize, 0.5, cam_color_map, 0.5, 2)

        result = np.zeros((172, 472, 3), dtype=np.uint8)
        result += 255
        result[22:150, 22:150, :] = cam_color_map
        result[22:150, 172:300, :] = x_resize
        result[22:150, 322:450, :] = add_image

        cv2.imshow("result", result)
        k = cv2.waitKey(-1)
        if k == ord('q'):
            exit()

        ### matplotlibバージョン
        # plt.subplot(1, 3, 1)
        # plt.imshow(cam_img)
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(x, cmap="gray")
        #
        # plt.subplot(1, 3, 3)
        # plt.imshow(x + cam_img)
        # plt.show()
