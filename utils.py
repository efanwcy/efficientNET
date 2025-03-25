import os
import sys
import torch
from tqdm import tqdm


def read_data(root: str):
    root_train = root + '/train'
    root_val = root + '/val'
    assert os.path.exists(root_train), "dataset root_train: {} does not exist.".format(root_train)
    assert os.path.exists(root_val), "dataset root_val: {} does not exist.".format(root_val)

    # 遍历训练文件夹，其下：一个文件夹对应一个类别
    #   ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    flower_class = [cla for cla in os.listdir(root_train) if os.path.isdir(os.path.join(root_train, cla))]

    # 生成类别名称以及对应的数字索引
    #   {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    train_images_path = []      # 存储训练集的所有图片路径
    train_images_label = []     # 存储训练集图片对应索引信息
    val_images_path = []        # 存储验证集的所有图片路径
    val_images_label = []       # 存储验证集图片对应索引信息
    every_class_train_num = []  # 存储每个类别的训练样本总数
    every_class_val_num = []    # 存储每个类别的验证样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_train_path = os.path.join(root_train, cla)
        # 遍历获取supported支持的所有文件路径
        #   os.path.splitext：分离文件名与扩展名；默认返回(fname,fextension)元组
        images_train = [os.path.join(root_train, cla, i) for i in os.listdir(cla_train_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引，image_class：0~4
        image_class = class_indices[cla]
        # 记录该类别的训练样本数量
        every_class_train_num.append(len(images_train))
        train_images_path += images_train
        train_images_label += [image_class] * len(images_train)

        cla_val_path = os.path.join(root_val, cla)
        images_val = [os.path.join(root_val, cla, i) for i in os.listdir(cla_val_path)
                  if os.path.splitext(i)[-1] in supported]
        every_class_val_num.append(len(images_val))
        val_images_path += images_val
        val_images_label += [image_class] * len(images_val)

    print("{} images were found in the train dataset.".format(sum(every_class_train_num)))
    print("{} images were found in the val dataset.".format(sum(every_class_val_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num
