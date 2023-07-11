import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
import os
import glob
# 加载DINOv2预训练权重
model =torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 定义文件夹路径
folder_path = '/home/lenovo/dinov2-1/dinov2-main/animals'

# 构建搜索模式
search_pattern = os.path.join(folder_path, '*.jpg')

# 获取所有匹配的文件路径
file_paths = glob.glob(search_pattern)
# 图像列表
images = []
image_paths=[]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 逐个加载图像并转换为张量
for path in file_paths:
    image = Image.open(path)
    image_tensor = transform(image)  # 使用 ToTensor() 转换为张量
    image_tensor = image_tensor.to(device)
    # 使用DINOv2提取特征
    with torch.no_grad():
        features = model(image_tensor.unsqueeze(0))
    images.append(features)
    image_paths.append(path[-5:])

print(len(images))

def similar(path):
    image = Image.open(path)
    image = image.resize((518, 518))  # 调整图像大小
    image = image.convert("RGB")  # 转换为RGB模式
    image_tensor = transform(image)  # 使用 ToTensor() 转换为张量
    image_tensor = image_tensor.to(device)
    output = model(image_tensor.unsqueeze(0))  
    test_vector=output
    similarities=[]

    for vector, path in zip(images, image_paths):
        similarity = 1 - cosine(test_vector.flatten().detach().cpu().numpy(), vector.flatten().detach().cpu().numpy())
        similarities.append((similarity.item(), path))

    # 排序相似度列表
    sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

    # 输出排序结果为路径
    for similarity, path in sorted_similarities:
        print(f"Similarity: {similarity}, Path: {path}")
    return
similar('/home/lenovo/dinov2-1/dinov2-main/animals/10.jpg')