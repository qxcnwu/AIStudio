import numpy as np
from PIL import Image
import paddle as pd
from paddle.vision.transforms import Normalize
import os


def Max_Min(data):
    """
    最大值最小值标准化
    :param data:
    :return:
    """
    x, y, z = data.shape

    def max_min(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    for i in range(z):
        data[:, :, i] = max_min(data[:, :, i])
    return data


def Z_score(data):
    """
    Z_score标准化
    :param data:
    :return:
    """
    x, y, z = data.shape

    def z_score(data):
        return (data - np.mean(data)) / np.maximum(np.std(data), 1 / 256)

    for i in range(z):
        data[:, :, i] = z_score(data[:, :, i])
    return data


def visio_pic(data, save_path):
    """
    可视化
    :param data:
    :param save_path:
    :return:
    """
    data *= 255
    img = Image.fromarray(data.astype('uint8'))
    img.save(save_path)
    return data


def norm_main(pic_path, func, save_path):
    img = np.array(Image.open(pic_path))
    data = visio_pic(func(img), save_path)
    return data


def paddle(path, save_path):
    normalize = Normalize(mean=[127.5, 127.5, 127.5],
                          std=[127.5, 127.5, 127.5],
                          data_format='HWC')

    fake_img = Image.open(path)
    fake_img = normalize(fake_img)
    out_img=fake_img
    fake_img=1/(1+np.exp(-fake_img))
    fake_img = ((fake_img - np.min(fake_img)) / ((np.max(fake_img) - np.min(fake_img)) / 256))
    fake_img = np.array(fake_img, dtype='int')
    im = Image.fromarray(fake_img.astype("uint8"))
    im.save(save_path)
    return fake_img,out_img

def sigmoid(path, save_path):
    fake_img = np.array(Image.open(path))
    fake_img=1/(1+np.exp(-((fake_img-127.5)/127.5)))
    out_img = fake_img
    fake_img = ((fake_img - np.min(fake_img)) / ((np.max(fake_img) - np.min(fake_img))/ 256))
    # print(np.max(fake_img) - np.min(fake_img))
    # fake_img = ((fake_img - np.min(fake_img)) / (0.1 / 256))
    fake_img = np.array(fake_img, dtype='int')
    im = Image.fromarray(fake_img.astype("uint8"))
    im.save(save_path)
    return fake_img,out_img

def channel_multi(pic_path, img_path):
    data = np.array(Image.open(pic_path))
    H, W, C = data.shape
    newdata = np.zeros(shape=[H, W, C])
    newdata[:, :, 2] = data[:, :, 0] / np.maximum(data[:, :, 1], 1)
    newdata[:, :, 0] = data[:, :, 2] / np.maximum(data[:, :, 1], 1)
    newdata[:, :, 1] = data[:, :, 0] / np.maximum(data[:, :, 2], 1)

    print(np.max(newdata[:, :, 0]) - np.min(newdata[:, :, 0]))
    print(np.max(newdata[:, :, 1]) - np.min(newdata[:, :, 1]))
    print(np.max(newdata[:, :, 2]) - np.min(newdata[:, :, 2]))

    newdata[:, :, 0] = ((newdata[:, :, 0] - np.min(newdata[:, :, 0])) / (
            (np.max(newdata[:, :, 0]) - np.min(newdata[:, :, 0])) / 256))
    newdata[:, :, 1] = ((newdata[:, :, 1] - np.min(newdata[:, :, 1])) / (
            (np.max(newdata[:, :, 1]) - np.min(newdata[:, :, 1])) / 256))
    newdata[:, :, 2] = ((newdata[:, :, 2] - np.min(newdata[:, :, 2])) / (
            (np.max(newdata[:, :, 2]) - np.min(newdata[:, :, 2])) / 256))

    fake_img = np.array(newdata, dtype='int')
    im = Image.fromarray(fake_img.astype("uint8"))
    im.save(img_path)
    return

if __name__ == '__main__':
    pic_name = "T000002.jpg"
    pic_path = "D:/xiazai/GOOGLE/train_and_label/img_train/" + pic_name
    func = "sigmoid"
    save_path = "result/" + func + "_" + pic_name
    data = sigmoid(pic_path, save_path)
    # data=norm_main(pic_path,Z_score,save_path)
