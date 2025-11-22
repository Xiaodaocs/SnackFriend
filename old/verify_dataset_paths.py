#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据集路径是否正确
"""

import os
import yaml

def verify_dataset_paths():
    """验证数据集路径是否正确"""
    print("=" * 60)
    print("验证数据集路径是否正确")
    print("=" * 60)
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, 'fastfood_yolo_dataset', 'data.yaml')
    
    print(f"数据集配置文件路径: {data_yaml_path}")
    
    # 检查配置文件是否存在
    if not os.path.exists(data_yaml_path):
        print(f"❌ 错误: 找不到数据集配置文件 {data_yaml_path}")
        return False
    
    # 读取配置文件
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        print("✅ 成功读取数据集配置文件")
    except Exception as e:
        print(f"❌ 错误: 读取数据集配置文件失败: {e}")
        return False
    
    # 检查配置项
    if 'train' not in data_config or 'val' not in data_config:
        print("❌ 错误: 配置文件中缺少train或val路径")
        return False
    
    # 获取训练和验证路径
    train_path = data_config['train']
    val_path = data_config['val']
    
    print(f"训练路径配置: {train_path}")
    print(f"验证路径配置: {val_path}")
    
    # 将相对路径转换为绝对路径
    data_dir = os.path.dirname(data_yaml_path)
    train_abs_path = os.path.join(data_dir, train_path)
    val_abs_path = os.path.join(data_dir, val_path)
    
    print(f"训练路径绝对路径: {train_abs_path}")
    print(f"验证路径绝对路径: {val_abs_path}")
    
    # 检查路径是否存在
    if not os.path.exists(train_abs_path):
        print(f"❌ 错误: 训练路径不存在 {train_abs_path}")
        return False
    
    if not os.path.exists(val_abs_path):
        print(f"❌ 错误: 验证路径不存在 {val_abs_path}")
        return False
    
    # 检查路径中是否包含文件
    train_files = os.listdir(train_abs_path)
    val_files = os.listdir(val_abs_path)
    
    if not train_files:
        print(f"❌ 错误: 训练路径中没有文件 {train_abs_path}")
        return False
    
    if not val_files:
        print(f"❌ 错误: 验证路径中没有文件 {val_abs_path}")
        return False
    
    print(f"✅ 训练路径包含 {len(train_files)} 个文件")
    print(f"✅ 验证路径包含 {len(val_files)} 个文件")
    
    # 检查是否有重复的目录名问题
    if 'fastfood_yolo_dataset' in train_path or 'fastfood_yolo_dataset' in val_path:
        print("⚠️ 警告: 路径中可能包含重复的目录名")
    
    print("✅ 所有路径验证通过")
    return True

if __name__ == "__main__":
    success = verify_dataset_paths()
    if success:
        print("\n✅ 路径验证通过! 数据集路径配置正确")
    else:
        print("\n❌ 路径验证失败! 数据集路径配置存在问题")