import os
import random
import json
import shutil       # 用于复制图片

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    #   ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    #   {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json.dumps:将一个Python数据结构转换为JSON，生成的是字符串
    #   indent:参数根据数据格式缩进显示，读起来更加清晰。
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # 会自动新建class_indices.json文件，往里面写入内容
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        #   os.path.splitext：分离文件名与扩展名；默认返回(fname,fextension)元组
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # 按比例随机采样验证样本
        #   random.sample：返回 k 长度从序列imagess中选择的新元素列表
        val_path = random.sample(images, k=int(len(images) * val_rate))

        new_val_path = 'D:/DeepLearning/classification/efficientNet/data/val/{0}'.format(cla)
        if os.path.exists(new_val_path) is False:
            os.makedirs(new_val_path)
        new_train_path = 'D:/DeepLearning/classification/efficientNet/data/train/{0}'.format(cla)
        if os.path.exists(new_train_path) is False:
            os.makedirs(new_train_path)

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                shutil.copy(img_path, new_val_path)
            else:  # 否则存入训练集
                shutil.copy(img_path, new_train_path)

if __name__ =="__main__":
    read_split_data(root="D:/DeepLearning/classification/efficientNet/data/flower_photos", val_rate=0.2)