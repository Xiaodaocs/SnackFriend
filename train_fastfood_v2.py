"""
使用完整数据集重新训练YOLO模型
包含10种食物类别：Baked Potato, Burger, Crispy Chicken, Donut, Fries, Hot Dog, Pizza, Sandwich, Taco, Taquito
"""

import os
from ultralytics import YOLO

def main():
    # 配置参数
    dataset_config = r"fastfood_v2_data.yaml"
    model_name = "models/yolov8m.pt"  # 使用medium版本，更准确
    epochs = 20  # 训练轮数（减少以加快训练速度）
    batch_size = 8  # 批次大小（减少以适应CPU内存）
    img_size = 640  # 图像大小
    project = "fastfood_v2_classification"  # 项目名称
    name = "train"  # 运行名称
    
    # 创建模型
    print(f"加载YOLO模型: {model_name}")
    model = YOLO(model_name)
    
    # 开始训练
    print("开始训练模型...")
    print(f"数据集配置: {dataset_config}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像大小: {img_size}")
    
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        save=True,  # 保存训练结果
        plots=True,  # 保存训练图表
        device='cpu',  # 使用CPU，因为没有GPU
        exist_ok=True,  # 允许覆盖已存在的项目
        pretrained=True,  # 使用预训练权重
        optimizer='Adam',  # 优化器
        lr0=0.001,  # 初始学习率
        patience=20,  # 早停耐心值
        save_period=5,  # 每5轮保存一次模型
        val=True,  # 验证
        amp=True,  # 自动混合精度
    )
    
    # 验证模型
    print("\n验证模型...")
    metrics = model.val()
    
    # 打印训练结果
    print("\n训练完成！")
    print(f"最佳模型保存在: {results.save_dir}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    # 导出模型
    print("\n导出模型...")
    model.export(format='onnx')  # 导出为ONNX格式
    
    print("全部完成！")

if __name__ == "__main__":
    main()