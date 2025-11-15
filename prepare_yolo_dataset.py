"""
准备YOLO格式的数据集目录结构
将原始数据集转换为YOLO训练所需的格式
"""

import os
import shutil
from pathlib import Path

# 定义路径
dataset_root = r"datasets\utkarshsaxenadn\fast-food-classification-dataset\versions\2\Fast Food Classification V2"
    yolo_dataset_root = r"datasets\fastfood_v2_yolo"

# 食物类别列表
food_categories = [
    "Baked Potato",
    "Burger", 
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Hot Dog",
    "Pizza",
    "Sandwich",
    "Taco",
    "Taquito"
]

# 创建YOLO数据集目录结构
print("创建YOLO数据集目录结构...")
os.makedirs(os.path.join(yolo_dataset_root, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_root, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_root, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_root, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_root, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(yolo_dataset_root, "labels", "test"), exist_ok=True)

# 处理训练集
print("处理训练集...")
train_dir = os.path.join(dataset_root, "Train")
for category_id, category in enumerate(food_categories):
    category_dir = os.path.join(train_dir, category)
    if os.path.exists(category_dir):
        # 复制图像文件
        for img_file in os.listdir(category_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(category_dir, img_file)
                dst_path = os.path.join(yolo_dataset_root, "images", "train", f"{category}_{img_file}")
                shutil.copy2(src_path, dst_path)
                
                # 创建对应的标签文件（假设整个图像是目标）
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                label_path = os.path.join(yolo_dataset_root, "labels", "train", f"{category}_{label_file}")
                
                # YOLO格式: class_id x_center y_center width height (归一化到0-1)
                # 这里假设整个图像是目标，所以边界框是整个图像
                with open(label_path, 'w') as f:
                    f.write(f"{category_id} 0.5 0.5 1.0 1.0\n")

# 处理验证集
print("处理验证集...")
val_dir = os.path.join(dataset_root, "Valid")
if os.path.exists(val_dir):
    for category_id, category in enumerate(food_categories):
        category_dir = os.path.join(val_dir, category)
        if os.path.exists(category_dir):
            # 复制图像文件
            for img_file in os.listdir(category_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(category_dir, img_file)
                    dst_path = os.path.join(yolo_dataset_root, "images", "val", f"{category}_{img_file}")
                    shutil.copy2(src_path, dst_path)
                    
                    # 创建对应的标签文件
                    label_file = img_file.rsplit('.', 1)[0] + '.txt'
                    label_path = os.path.join(yolo_dataset_root, "labels", "val", f"{category}_{label_file}")
                    
                    with open(label_path, 'w') as f:
                        f.write(f"{category_id} 0.5 0.5 1.0 1.0\n")

# 处理测试集
print("处理测试集...")
test_dir = os.path.join(dataset_root, "Test")
if os.path.exists(test_dir):
    for category_id, category in enumerate(food_categories):
        category_dir = os.path.join(test_dir, category)
        if os.path.exists(category_dir):
            # 复制图像文件
            for img_file in os.listdir(category_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_path = os.path.join(category_dir, img_file)
                    dst_path = os.path.join(yolo_dataset_root, "images", "test", f"{category}_{img_file}")
                    shutil.copy2(src_path, dst_path)
                    
                    # 创建对应的标签文件
                    label_file = img_file.rsplit('.', 1)[0] + '.txt'
                    label_path = os.path.join(yolo_dataset_root, "labels", "test", f"{category}_{label_file}")
                    
                    with open(label_path, 'w') as f:
                        f.write(f"{category_id} 0.5 0.5 1.0 1.0\n")

print("YOLO数据集准备完成！")
print(f"数据集保存在: {yolo_dataset_root}")

# 统计各类别图像数量
print("\n图像数量统计:")
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(yolo_dataset_root, "images", split)
    if os.path.exists(img_dir):
        img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{split}: {img_count} 张图像")