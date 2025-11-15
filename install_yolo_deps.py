# -*- coding: utf-8 -*-
"""
YOLO依赖库安装脚本
用于安装运行YOLO程序所需的依赖库
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装 {package} 失败: {e}")
        return False

def main():
    print("===== YOLO依赖库安装脚本 =====")
    print("此脚本将安装运行YOLO程序所需的依赖库")
    print("需要的库包括: ultralytics, opencv-python, pyyaml, matplotlib, pandas, pillow")
    print()
    
    # 确认是否继续
    while True:
        choice = input("是否继续安装依赖库? (y/n): ")
        if choice.lower() == 'y':
            break
        elif choice.lower() == 'n':
            print("安装已取消")
            return
        else:
            print("无效输入，请输入 'y' 或 'n'")
    
    # 需要安装的包列表
    packages = [
        "ultralytics",      # YOLOv8实现
        "opencv-python",    # 图像处理
        "pyyaml",          # YAML文件处理
        "matplotlib",      # 数据可视化
        "pandas",          # 数据处理
        "pillow",          # 图像处理
        "numpy",           # 数值计算
        "seaborn"          # 数据可视化
    ]
    
    # 安装每个包
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print()
    print(f"安装完成! 成功安装 {success_count}/{len(packages)} 个包")
    
    if success_count == len(packages):
        print("所有依赖库安装成功，现在可以运行YOLO程序了!")
        print("运行命令: python yolo_main.py")
    else:
        print("部分依赖库安装失败，请检查错误信息并手动安装失败的包")
        print("您也可以使用以下命令手动安装:")
        for package in packages:
            print(f"pip install {package}")

if __name__ == "__main__":
    main()