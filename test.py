import os
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

    # 读取 clean 文件夹中的图片
    clean_folder_path = "./test"  # clean 文件夹路径
    if not os.path.exists(clean_folder_path):
        raise FileNotFoundError(f"未找到 clean 文件夹: {clean_folder_path}")

    # 获取 clean 文件夹中的所有图片路径
    clean_images = [os.path.join(clean_folder_path, img) for img in os.listdir(clean_folder_path) if img.endswith(('.jpg', '.png'))]

    # 统计预测结果
    clean_wound_count = 0  # 预测为 clean_wound 的次数
    contamination_wound_count = 0  # 预测为 contamination_wound 的次数

    for img_path in clean_images:
        # 预处理图像
        img_tensor = preprocess_image(img_path, data_transform).to(device)

        # 模型推理
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)  # 获取预测类别
            predicted_label = "clean_wound" if predicted.item() == 0 else "contamination_wound"

        # 统计结果
        if predicted_label == "clean_wound":
            clean_wound_count += 1
        else:
            contamination_wound_count += 1

    # 输出结果
    print(f"预测为 clean_wound 的次数: {clean_wound_count}")
    print(f"预测为 contamination_wound 的次数: {contamination_wound_count}")


if __name__ == '__main__':
    main()