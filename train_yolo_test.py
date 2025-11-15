#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用修复后的数据集配置进行YOLO训练
"""

import os
import sys
from ultralytics import YOLO

def train_yolo_model():
    """使用修复后的数据集配置进行YOLO训练"""
    print("=" * 60)
    print("使用修复后的数据集配置进行YOLO训练")
    print("=" * 60)
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'fastfood_yolo_dataset', 'data.yaml')
    
    print(f"数据集配置文件路径: {data_yaml_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"❌ 错误: 找不到数据集配置文件 {data_yaml_path}")
        return False
    
    # 创建YOLO模型
    try:
        print("正在创建YOLO模型...")
        model = YOLO('yolov8n.pt')  # 使用预训练的YOLOv8n模型
        print("✅ 成功创建YOLO模型")
    except Exception as e:
        print(f"❌ 错误: 创建YOLO模型失败: {e}")
        return False
    
    # 使用数据集配置文件进行训练
    try:
        print("正在开始训练...")
        # 使用修复后的数据集配置进行训练
        # 设置epochs=3进行短暂训练，仅用于验证
        results = model.train(
            data=data_yaml_path,
            epochs=3,
            imgsz=640,
            batch=16,
            name='fastfood_yolo_test',
            project='runs/detect',
            exist_ok=True,
            verbose=True
        )
        print("✅ 训练成功完成")
        return True
    except Exception as e:
        print(f"❌ 错误: 训练失败: {e}")
        return False

if __name__ == "__main__":
    success = train_yolo_model()
    if success:
        print("\n✅ 训练测试通过! 数据集配置修复成功")
    else:
        print("\n❌ 训练测试失败! 数据集配置仍有问题")
    
    sys.exit(0 if success else 1)