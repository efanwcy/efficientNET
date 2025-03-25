import os
import random
import torch
from torchvision import transforms
from PIL import Image
from model import efficientnet_b0 as create_model


def load_model(model_path, num_classes, device):
    """
    加载训练好的模型
    :param model_path: 模型权重路径
    :param num_classes: 类别数
    :param device: 设备（CPU 或 GPU）
    :return: 加载好的模型
    """
    model = create_model(num_classes=num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 设置 weights_only=True
    else:
        raise FileNotFoundError("未找到模型权重文件: {}".format(model_path))
    model.eval()  # 设置为评估模式
    return model


def preprocess_image(image_path, transform):
    """
    预处理图像
    :param image_path: 图像路径
    :param transform: 图像预处理方法
    :return: 预处理后的图像张量
    """
    img = Image.open(image_path)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)  # 增加 batch 维度
    return img


def main():
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    img_size = 224  # EfficientNet-B0 的输入尺寸
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载模型
    model_path = "./output/model-29.pth"  # 模型权重路径
    num_classes = 2  # 类别数
    model = load_model(model_path, num_classes, device)

    # 读取验证集数据
    val_data_path = "./data/test"  # 验证集路径
    class_names = ["clean_wound", "contamination_wound"]  # 类别名称
    image_paths = []  # 存储所有图像路径
    labels = []  # 存储所有图像标签

    for class_name in class_names:
        class_path = os.path.join(val_data_path, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"未找到类别文件夹: {class_path}")
        # 获取该类别下的所有图像路径
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(('.jpg', '.png'))]
        image_paths.extend(class_images)
        labels.extend([class_name] * len(class_images))

    # 随机选择 100 张图片
    sample_size = min(100, len(image_paths))  # 不超过验证集的总图片数量
    selected_indices = random.sample(range(len(image_paths)), sample_size)
    selected_images = [image_paths[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    # 统计正确预测次数
    correct_total = 0  # 总正确预测次数

    for img_path, true_label in zip(selected_images, selected_labels):
        # 预处理图像
        img_tensor = preprocess_image(img_path, data_transform).to(device)

        # 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)  # 获取预测类别
            predicted_label = class_names[predicted.item()]

        # 统计结果
        if predicted_label == true_label:
            correct_total += 1

    # 输出总正确预测次数
    print(f"总正确预测次数: {correct_total}")


if __name__ == '__main__':
    main()