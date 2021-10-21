import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Transpose, ColorJitter, vflip, Grayscale, hflip, Normalize
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from paddleseg.models.backbones.hrnet import HRNet_W64
from paddleseg.models.ocrnet import OCRNet

def predict(model_path="",pic_dir="",save_dir=""):
    files=os.listdir(pic_dir)
    model = OCRNet(5, HRNet_W64(pretrained=None), (0,))
    layer_state_dict = paddle.load(model_path)
    model.set_state_dict(layer_state_dict)
    model.eval()
    for i in tqdm(files):
        path=os.path.join(pic_dir,i)
        data=np.array(Image.open(path))
        data = 1 / (1 + np.exp(-((data - 127.5) / 127.5)))
        pic_data = paddle.transpose(paddle.cast(paddle.to_tensor(data), dtype='float32'), [0, 3, 1, 2])
        predicts = model(pic_data)
        pic = paddle.transpose(predicts[0], [0, 2, 3, 1])
        ans = paddle.argmax(pic, axis=-1)
        im=Image.fromarray(ans.astype("uint8"))
        im.save(os.path.join(save_dir,i.replace("jpg","png")))
    return

if __name__ == '__main__':
    predict()