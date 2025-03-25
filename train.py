import os
import math
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
from utils import read_data, train_one_epoch, evaluate


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    print(args)

    if os.path.exists("./output") is False:
        os.makedirs("./output")

    # 获取训练与验证图片路径及标签，均是列表形式
    train_images_path, train_images_label, val_images_path, val_images_label = read_data(args.data_path)

    # B0~B7图像输入尺寸不同，会影响下面一些参数
    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    # train和val预处理函数
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                     transforms.RandomHorizontalFlip(),     # 水平方向随机翻转
                                     transforms.ToTensor(),     # 转化成tensor，数值从0~255，变成0~1
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # mean std
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.CenterCrop(img_size[num_model]),
                                   transforms.ToTensor(),       
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}    

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 实例化模型
    model = create_model(num_classes=args.num_classes).to(device)

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            # -----------------------------------------------------------------------------------------#
            # 由于预训练权重是在Imagenet上的，类别数1000，而花分类数据集只有5类，故此处分类器权重不进行加载
            #   训练过程中会出现如下提示，正常，不用管
            #   _IncompatibleKeys(missing_keys=['classifier.1.weight', 'classifier.1.bias'], unexpected_keys=[])
            # -----------------------------------------------------------------------------------------#
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        torch.save(model.state_dict(), "./output/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 数据集所在目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./data")
    # download model weights
    parser.add_argument('--weights', type=str, default='./pretrained/efficientnetb0.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    
    opt = parser.parse_args()

    main(opt)
