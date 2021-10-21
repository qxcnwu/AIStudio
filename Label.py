from PIL import Image
import numpy as np
import os
import random

def make_label_pic(label_dir,test_save,train_save):
    files=os.listdir(label_dir)
    num=int(len(files)*0.8)
    files=list(files)
    random.shuffle(files)
    train_list=files[0:num]
    test_list=files[num:-1]

    file=open(train_save,"w")
    for i in train_list:
        file.write(i+","+i.replace(".jpg",".png")+"\n")
    file.close()

    file = open(test_save, "w")
    for i in test_list:
        file.write(i + "," + i.replace(".jpg", ".png") + "\n")
    file.close()

    return

if __name__ == '__main__':
    label_dir="D:/xiazai/GOOGLE/train_and_label/img_train/"
    test_save="test.txt"
    train_save="train.txt"
    make_label_pic(label_dir,test_save,train_save)

