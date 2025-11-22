# -*- coding: utf-8 -*-
"""
YOLO快餐图像分类测试程序 - 最终版
使用预训练模型进行测试，不进行训练
"""

import os
import cv2
import numpy as np
import random
from ultralytics import YOLO

def create_simple_food_image(food_type, image_size=(640, 640)):
    """
    创建简单的食物图像
    food_type: 食物类型 ('Burger', 'Pizza', 'Donut')
    image_size: 图像尺寸 (width, height)
    """
    width, height = image_size
    # 创建白色背景
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 根据食物类型创建不同的简单形状
    if food_type == 'Burger':
        # 创建汉堡 - 棕色圆形
        center = (width // 2, height // 2)
        radius = min(width, height) // 4
        cv2.circle(img, center, radius, (139, 69, 19), -1)  # 棕色
        # 添加一些细节
        cv2.circle(img, center, radius - 20, (255, 215, 0), -1)  # 金黄色
        cv2.circle(img, center, radius - 40, (139, 69, 19), -1)  # 棕色
        
    elif food_type == 'Pizza':
        # 创建披萨 - 红色三角形
        points = np.array([
            [width // 2, height // 4],
            [width // 4, 3 * height // 4],
            [3 * width // 4, 3 * height // 4]
        ], np.int32)
        cv2.fillPoly(img, [points], (255, 0, 0))  # 红色
        # 添加一些细节
        center = (width // 2, height // 2)
        cv2.circle(img, center, 30, (255, 255, 0), -1)  # 黄色中心
        
    elif food_type == 'Donut':
        # 创建甜甜圈 - 粉色圆环
        center = (width // 2, height // 2)
        outer_radius = min(width, height) // 4
        inner_radius = outer_radius // 2
        cv2.circle(img, center, outer_radius, (255, 192, 203), -1)  # 粉色
        cv2.circle(img, center, inner_radius, (255, 255, 255), -1)  # 白色中心
        # 添加一些糖霜点
        for _ in range(10):
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(inner_radius + 10, outer_radius - 10)
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            cv2.circle(img, (x, y), 5, (255, 0, 255), -1)  # 紫色糖霜
    
    # 添加一些随机噪声，使图像更真实
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def create_test_dataset(food_types=['Burger', 'Pizza', 'Donut'], num_images=5):
    """
    创建测试数据集
    food_types: 食物类型列表
    num_images: 每种类型的图片数量
    """
    # 创建测试数据目录
    test_dir = 'test_data_final'
    os.makedirs(test_dir, exist_ok=True)
    
    for food_type in food_types:
        # 为每种食物类型创建目录
        food_dir = os.path.join(test_dir, food_type)
        os.makedirs(food_dir, exist_ok=True)
        
        # 生成图片
        for i in range(1, num_images + 1):
            img = create_simple_food_image(food_type)
            img_path = os.path.join(food_dir, f'{food_type}_{i}.jpg')
            cv2.imwrite(img_path, img)
            print(f'已生成: {img_path}')
    
    print(f'测试数据集已创建在 {test_dir} 目录')
    return test_dir

def test_pretrained_model(test_dir, classes, model_path='yolov8n.pt', conf_threshold=0.25):
    """
    使用预训练模型测试
    test_dir: 测试数据目录
    classes: 类别列表
    model_path: 模型路径
    conf_threshold: 置信度阈值
    """
    # 加载预训练模型
    print(f"加载预训练模型: {model_path}")
    model = YOLO(model_path)
    
    # 获取测试图像
    test_images = []
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.endswith('.jpg') or f.endswith('.png')]
        test_images.extend([(img, class_name) for img in images])
    
    # 随机选择几张图像进行测试
    random.shuffle(test_images)
    test_images = test_images[:6]  # 测试6张图像
    
    # 创建输出目录
    os.makedirs('test_results_final', exist_ok=True)
    
    # 打印模型可检测的类别
    print("\n模型可检测的类别:")
    for i, name in enumerate(model.names):
        print(f"  {i}: {name}")
    
    # 检查我们的类别是否在模型中
    our_classes_in_model = []
    for our_class in classes:
        for model_class_id, model_class_name in model.names.items():
            if our_class.lower() in model_class_name.lower() or model_class_name.lower() in our_class.lower():
                our_classes_in_model.append((our_class, model_class_id, model_class_name))
                break
    
    print(f"\n我们的类别在模型中的匹配:")
    for our_class, model_id, model_name in our_classes_in_model:
        print(f"  {our_class} -> {model_name} (ID: {model_id})")
    
    for i, (img_path, true_class) in enumerate(test_images):
        # 读取图像
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 进行预测
        results = model(img_rgb, conf=conf_threshold)
        
        # 绘制检测结果
        result_img = img.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for j, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                pred_class = model.names[class_id]
                
                # 绘制边界框
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 添加标签
                label = f'{pred_class}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                print(f"图像: {os.path.basename(img_path)}, 真实类别: {true_class}, 预测类别: {pred_class}, 置信度: {conf:.2f}")
        else:
            cv2.putText(result_img, '未检测到任何目标', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"图像: {os.path.basename(img_path)}, 真实类别: {true_class}, 预测: 未检测到任何目标")
        
        # 保存结果图像
        result_path = os.path.join('test_results_final', f'result_{i}_{os.path.basename(img_path)}')
        cv2.imwrite(result_path, result_img)
    
    print(f"测试完成，结果已保存到 test_results_final 目录")

def test_trained_model(test_dir, classes, model_path='runs/detect/fastfood_classification_test/weights/best.pt', conf_threshold=0.25):
    """
    使用训练好的模型测试
    test_dir: 测试数据目录
    classes: 类别列表
    model_path: 模型路径
    conf_threshold: 置信度阈值
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    # 加载训练好的模型
    print(f"加载训练好的模型: {model_path}")
    model = YOLO(model_path)
    
    # 获取测试图像
    test_images = []
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.endswith('.jpg') or f.endswith('.png')]
        test_images.extend([(img, class_name) for img in images])
    
    # 随机选择几张图像进行测试
    random.shuffle(test_images)
    test_images = test_images[:6]  # 测试6张图像
    
    # 创建输出目录
    os.makedirs('test_results_trained', exist_ok=True)
    
    # 打印模型可检测的类别
    print("\n模型可检测的类别:")
    for i, name in enumerate(model.names):
        print(f"  {i}: {name}")
    
    for i, (img_path, true_class) in enumerate(test_images):
        # 读取图像
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 进行预测
        results = model(img_rgb, conf=conf_threshold)
        
        # 绘制检测结果
        result_img = img.copy()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for j, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box.astype(int)
                pred_class = model.names[class_id]
                
                # 绘制边界框
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 添加标签
                label = f'{pred_class}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                print(f"图像: {os.path.basename(img_path)}, 真实类别: {true_class}, 预测类别: {pred_class}, 置信度: {conf:.2f}")
        else:
            cv2.putText(result_img, '未检测到任何目标', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"图像: {os.path.basename(img_path)}, 真实类别: {true_class}, 预测: 未检测到任何目标")
        
        # 保存结果图像
        result_path = os.path.join('test_results_trained', f'result_{i}_{os.path.basename(img_path)}')
        cv2.imwrite(result_path, result_img)
    
    print(f"测试完成，结果已保存到 test_results_trained 目录")

def main():
    """
    主函数
    """
    print("YOLO快餐图像分类程序 - 最终版")
    print("=" * 50)
    
    # 1. 创建测试数据集
    print("\n1. 创建测试数据集...")
    test_dir = create_test_dataset(num_images=5)  # 每个类别5张图片
    
    # 2. 使用预训练模型测试
    print("\n2. 使用预训练模型测试...")
    classes = ['Burger', 'Pizza', 'Donut']
    test_pretrained_model(test_dir, classes, conf_threshold=0.1)  # 降低置信度阈值
    
    # 3. 使用训练好的模型测试
    print("\n3. 使用训练好的模型测试...")
    test_trained_model(test_dir, classes, conf_threshold=0.1)  # 降低置信度阈值
    
    print("\nYOLO快餐图像分类程序测试完成!")

if __name__ == "__main__":
    main()