import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Transpose, ColorJitter, vflip, Grayscale, hflip, Normalize
from PIL import Image
import numpy as np
import random
import os


# 读取数据集
class Dataloader(Dataset):
    def __init__(self, model=''):
        super(Dataloader).__init__()
        self.image_floader = 'D:/xiazai/GOOGLE/train_and_label/'
        if model == 'train':
            self.file = 'train.txt'
        else:
            self.file = 'test.txt'
        self.jpg_list, self.label_list= self.read_list()

    def read_list(self):
        data_list = []
        jpg_list = []
        label_list = []
        with open(self.file) as lines:
            for line in lines:
                jpg_path = os.path.join(self.image_floader+"/img_train/", line.split(',')[0])
                label_path = os.path.join(self.image_floader+"/lab_train/", line.split(',')[1].replace('\n', ''))
                data_list.append((jpg_path, label_path))
        random.shuffle(data_list)
        for k in data_list:
            jpg_list.append(k[0])
            label_list.append(k[1])
        return jpg_list, label_list

    def _load_img(self, jpg_path, label_path):
        jpg = np.array(Image.open(jpg_path))
        jpg = 1 / (1 + np.exp(-((jpg - 127.5) / 127.5)))
        label = Image.open(label_path)
        return Compose([Transpose()])(jpg), Compose([Grayscale(), Transpose()])(label)

    def __getitem__(self, idx):
        train_image, label_image= self._load_img(self.jpg_list[idx], self.label_list[idx])
        train_image = np.array(train_image, dtype='float32')
        label_image = np.array(label_image, dtype='int64')
        label_image[label_image>4]=4
        zero_image=np.zeros(shape=label_image.shape)
        label_image=np.concatenate([label_image,zero_image])
        return train_image, label_image

    def __len__(self):
        return len(self.label_list)

# 计算损失函数
class LOSS_CROSS_IOU(paddle.nn.Layer):
    def __init__(self, weights, num_class):
        super(LOSS_CROSS_IOU, self).__init__()
        self.weights_list = weights
        self.num_class = num_class

    def forward(self, input, label):
        input_1 = paddle.transpose(input[1], [0, 2, 3, 1])
        input = paddle.transpose(input[0], [0, 2, 3, 1])
        label_1 = paddle.cast(paddle.transpose(paddle.unsqueeze(label[:,0,:,:],axis=1), [0, 2, 3, 1]),dtype="int64")
        iou_loss = paddle.abs(paddle.mean(paddle.nn.functional.dice_loss(input,label_1)))
        cross_loss = paddle.mean(paddle.nn.functional.softmax_with_cross_entropy(logits=input, label=label_1))+paddle.mean(paddle.nn.functional.softmax_with_cross_entropy(logits=input_1, label=label_1))
        return paddle.add(iou_loss, cross_loss)

def train():
    from paddleseg.models.backbones.hrnet import HRNet_W48
    from paddleseg.models.ocrnet import OCRNet
    NET=OCRNet(5,HRNet_W48(pretrained='https://bj.bcebos.com/paddleseg/dygraph/hrnet_w64_ssld.tar.gz'),(0,))

    epoch=20
    batch_size = 1
    step = 0
    step2 = 0
    PrintNum = 1
    modelDir = "ImageNet/Model/"
    ans_list=[]
    NET.train()

    # 训练测试数据集
    train_dataset = Dataloader(model='train')
    val_dataset = Dataloader(model='test')
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # 设置学习率，设置优化器
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.8, verbose=True)
    sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=NET.parameters())
    loss_fn = LOSS_CROSS_IOU(0, 5)
    for i in range(epoch):
        losses = []
        # 按Batch循环
        for batch_id, data in enumerate(train_loader()):
            step += 1
            x_data = data[0]  # 训练数据
            y_data = data[1]
            predicts = NET(x_data)
            loss = loss_fn(predicts, y_data)
            losses.append(loss.numpy())
            if batch_id % PrintNum == 0:
                print('AFTER ', i + 1, ' epochs', batch_id + 1, ' batch iou:', sum(losses) / len(losses))
            loss.backward()
            sgd.step()
            # 梯度清零
            sgd.clear_grad()
        scheduler.step()
        print('epoch iou:', sum(losses) / len(losses))
        NET.eval()
        with paddle.no_grad():
            aiou = []
            for batch_id, data in enumerate(val_loader()):
                step2 += 1
                x_data = data[0]  # 训练数据
                y_data = data[1]
                predicts = NET(x_data)
                loss = loss_fn(predicts, y_data)
                aiou.append(loss.numpy())
                if batch_id % PrintNum == 0:
                    print('test biou ', sum(aiou) / len(aiou))
            print('test biou all', sum(aiou) / len(aiou))
            ans_list.append(sum(aiou) / len(aiou))
        NET.train()
        if ans_list[-1] == min(ans_list):
            model_path = modelDir + str(i)
            paddle.save(NET.state_dict(), os.path.join(model_path, 'model.pdparams'))
            paddle.save(sgd.state_dict(), os.path.join(model_path, 'model.pdopt'))
        else:
            print(ans_list[-1], min(ans_list), 'no save')
    return 0

if __name__ == '__main__':
    train()