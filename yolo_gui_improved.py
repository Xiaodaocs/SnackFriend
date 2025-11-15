# -*- coding: utf-8 -*-
"""
YOLO快餐识别系统 - GUI版本 (改进版)
提供图形界面，用户可以选择训练模型或测试模型
修复了"开始训练"按钮问题，并支持使用已有的训练数据集
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import shutil
from ultralytics import YOLO
import yaml
import random

class FastFoodClassifierGUI:
    def __init__(self, root):
        """
        初始化GUI界面
        """
        self.root = root
        self.root.title("YOLO快餐识别系统")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 获取当前脚本所在目录，用于相对路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TButton', font=('微软雅黑', 10))
        self.style.configure('TLabel', font=('微软雅黑', 10))
        self.style.configure('TFrame', background='#f0f0f0')
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        title_label = ttk.Label(self.main_frame, text="YOLO快餐识别系统", font=('微软雅黑', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 创建按钮框架
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        # 训练模型按钮
        self.train_button = ttk.Button(button_frame, text="训练模型", command=self.open_training_window)
        self.train_button.pack(side=tk.LEFT, padx=20)
        
        # 测试模型按钮
        self.test_button = ttk.Button(button_frame, text="测试模型", command=self.open_testing_window)
        self.test_button.pack(side=tk.LEFT, padx=20)
        
        # 创建信息显示区域
        info_frame = ttk.LabelFrame(self.main_frame, text="系统信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, width=70, font=('微软雅黑', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 显示欢迎信息
        self.update_info("欢迎使用YOLO快餐识别系统！\n\n请选择操作：\n1. 点击'训练模型'按钮训练新的YOLO模型\n2. 点击'测试模型'按钮使用已有模型识别快餐图像\n\n注意：首次使用前请确保已安装所需的依赖库。")
        
        # 检查训练数据目录
        self.check_training_data()
        
    
    def update_info(self, message):
        """
        更新信息显示区域
        """
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_training_data(self):
        """
        检查训练数据目录是否存在
        """
        # 修改为使用相对路径，基于当前脚本所在目录
        self.training_data_dir = os.path.join(self.current_dir, "fastfood_yolo_dataset")
        if os.path.exists(self.training_data_dir):
            self.update_info("欢迎使用YOLO快餐识别系统！\n\n检测到训练数据目录，可以进行模型训练。\n\n请选择操作：\n1. 点击'训练模型'按钮训练新的YOLO模型\n2. 点击'测试模型'按钮使用已有模型识别快餐图像")
        else:
            self.update_info("欢迎使用YOLO快餐识别系统！\n\n未检测到训练数据目录，需要先准备训练数据。\n\n请选择操作：\n1. 点击'训练模型'按钮准备训练数据并训练模型\n2. 点击'测试模型'按钮使用已有模型识别快餐图像")
    
    def open_training_window(self):
        """
        打开训练模型窗口
        """
        training_window = tk.Toplevel(self.root)
        training_window.title("模型训练")
        training_window.geometry("700x600")
        training_window.resizable(True, True)
        
        # 创建训练框架
        training_frame = ttk.Frame(training_window, padding="10")
        training_frame.pack(fill=tk.BOTH, expand=True)
        
        # 训练参数框架
        params_frame = ttk.LabelFrame(training_frame, text="训练参数", padding="10")
        params_frame.pack(fill=tk.X, pady=10)
        
        # 数据目录
        ttk.Label(params_frame, text="训练数据目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_dir_var = tk.StringVar(value=self.training_data_dir)
        data_dir_entry = ttk.Entry(params_frame, textvariable=self.data_dir_var, width=40)
        data_dir_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        browse_button = ttk.Button(params_frame, text="浏览", command=lambda: self.browse_directory(self.data_dir_var))
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # 数据源选择
        ttk.Label(params_frame, text="数据源:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.data_source_var = tk.StringVar(value="create")
        data_source_frame = ttk.Frame(params_frame)
        data_source_frame.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Radiobutton(data_source_frame, text="创建示例数据", variable=self.data_source_var, value="create").pack(side=tk.LEFT)
        ttk.Radiobutton(data_source_frame, text="使用已有数据集", variable=self.data_source_var, value="existing").pack(side=tk.LEFT, padx=10)
        
        # 已有数据集路径 - 修改为相对路径
        ttk.Label(params_frame, text="已有数据集路径:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.existing_data_var = tk.StringVar(value=os.path.join(self.current_dir, "fastfood_yolo_dataset"))
        existing_data_entry = ttk.Entry(params_frame, textvariable=self.existing_data_var, width=40)
        existing_data_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        browse_existing_button = ttk.Button(params_frame, text="浏览", command=lambda: self.browse_directory(self.existing_data_var))
        browse_existing_button.grid(row=2, column=2, padx=5, pady=5)
        
        # 训练轮数
        ttk.Label(params_frame, text="训练轮数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="10")
        epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 图像尺寸
        ttk.Label(params_frame, text="图像尺寸:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.img_size_var = tk.StringVar(value="640")
        img_size_entry = ttk.Entry(params_frame, textvariable=self.img_size_var, width=10)
        img_size_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # 批次大小
        ttk.Label(params_frame, text="批次大小:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.StringVar(value="16")
        batch_size_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10)
        batch_size_entry.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # 模型选择 - 修改为相对路径
        ttk.Label(params_frame, text="预训练模型:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value=os.path.join(self.current_dir, "yolov8s.pt"))  # 修改默认模型为本地已有的模型
        model_combo = ttk.Combobox(params_frame, textvariable=self.model_var, width=20)
        model_combo['values'] = (
            os.path.join(self.current_dir, "yolov8n.pt"),
            os.path.join(self.current_dir, "yolov8s.pt"),
            os.path.join(self.current_dir, "yolov8m.pt"),
            os.path.join(self.current_dir, "yolov8l.pt"),
            os.path.join(self.current_dir, "yolov8x.pt")
        )
        model_combo.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # 训练信息显示区域
        info_frame = ttk.LabelFrame(training_frame, text="训练信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        training_info = scrolledtext.ScrolledText(info_frame, height=10, width=70, font=('微软雅黑', 9))
        training_info.pack(fill=tk.BOTH, expand=True)
        
        # 按钮框架
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(pady=10)
        
        # 准备数据按钮
        prepare_data_button = ttk.Button(button_frame, text="准备训练数据", 
                                        command=lambda: self.prepare_training_data(training_info))
        prepare_data_button.pack(side=tk.LEFT, padx=5)
        
        # 开始训练按钮 - 修复了按钮问题
        start_train_button = ttk.Button(button_frame, text="开始训练", 
                                       command=lambda: self.start_training(training_info))
        start_train_button.pack(side=tk.LEFT, padx=5)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=training_window.destroy)
        close_button.pack(side=tk.LEFT, padx=5)
        
        # 显示训练说明
        training_info.insert(tk.END, "模型训练说明:\n\n")
        training_info.insert(tk.END, "1. 准备训练数据: 选择数据源并准备训练数据\n")
        training_info.insert(tk.END, "2. 设置训练参数: 调整训练轮数、图像尺寸等参数\n")
        training_info.insert(tk.END, "3. 开始训练: 使用设置好的参数开始训练模型\n\n")
        training_info.insert(tk.END, "注意: 训练过程可能需要较长时间，请耐心等待。\n")
        training_info.insert(tk.END, "训练完成后，模型将保存在 runs/detect/fastfood_classification/weights/ 目录下。\n")
    
    def open_testing_window(self):
        """
        打开测试模型窗口
        """
        testing_window = tk.Toplevel(self.root)
        testing_window.title("模型测试")
        testing_window.geometry("800x600")
        testing_window.resizable(True, True)
        
        # 创建测试框架
        testing_frame = ttk.Frame(testing_window, padding="10")
        testing_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧框架 - 参数设置
        left_frame = ttk.Frame(testing_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 测试参数框架
        params_frame = ttk.LabelFrame(left_frame, text="测试参数", padding="10")
        params_frame.pack(fill=tk.X, pady=10)
        
        # 模型选择
        ttk.Label(params_frame, text="模型文件:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.test_model_var = tk.StringVar()
        
        # 查找可用的模型文件
        model_files = self.find_model_files()
        if model_files:
            self.test_model_var.set(model_files[0])
        
        model_combo = ttk.Combobox(params_frame, textvariable=self.test_model_var, width=40)
        model_combo['values'] = model_files
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        browse_button = ttk.Button(params_frame, text="浏览", command=lambda: self.browse_file(self.test_model_var))
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # 图像路径
        ttk.Label(params_frame, text="测试图像:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.image_path_var = tk.StringVar()
        image_path_entry = ttk.Entry(params_frame, textvariable=self.image_path_var, width=40)
        image_path_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        browse_image_button = ttk.Button(params_frame, text="浏览", command=lambda: self.browse_image_file())
        browse_image_button.grid(row=1, column=2, padx=5, pady=5)
        
        # 置信度阈值
        ttk.Label(params_frame, text="置信度阈值:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.conf_threshold_var = tk.StringVar(value="0.25")
        conf_threshold_entry = ttk.Entry(params_frame, textvariable=self.conf_threshold_var, width=10)
        conf_threshold_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 按钮框架
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)
        
        # 测试图像按钮
        test_button = ttk.Button(button_frame, text="测试图像", command=lambda: self.test_image(testing_info, result_frame))
        test_button.pack(side=tk.LEFT, padx=5)
        
        # 关闭按钮
        close_button = ttk.Button(button_frame, text="关闭", command=testing_window.destroy)
        close_button.pack(side=tk.LEFT, padx=5)
        
        # 测试信息显示区域
        info_frame = ttk.LabelFrame(left_frame, text="测试信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        testing_info = scrolledtext.ScrolledText(info_frame, height=10, width=40, font=('微软雅黑', 9))
        testing_info.pack(fill=tk.BOTH, expand=True)
        
        # 右侧框架 - 结果显示
        right_frame = ttk.Frame(testing_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(right_frame, text="检测结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建图像显示标签
        self.image_label = ttk.Label(result_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # 显示测试说明
        testing_info.insert(tk.END, "模型测试说明:\n\n")
        testing_info.insert(tk.END, "1. 选择模型文件: 选择要使用的训练好的模型\n")
        testing_info.insert(tk.END, "2. 选择测试图像: 选择要测试的图像文件\n")
        testing_info.insert(tk.END, "3. 设置置信度阈值: 调整检测的置信度阈值\n")
        testing_info.insert(tk.END, "4. 点击'测试图像'按钮开始检测\n\n")
        testing_info.insert(tk.END, "检测结果将在右侧显示，包括检测框和标签。\n")
    
    def browse_directory(self, var):
        """
        浏览目录
        """
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def browse_file(self, var):
        """
        浏览文件
        """
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("模型文件", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            var.set(file_path)
    
    def browse_image_file(self):
        """
        浏览图像文件
        """
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_path_var.set(file_path)
    
    def find_model_files(self):
        """
        查找可用的模型文件
        """
        model_files = []
        
        # 查找当前目录下的模型文件
        for file in os.listdir(self.current_dir):
            if file.endswith('.pt'):
                model_files.append(os.path.join(self.current_dir, file))
        
        # 查找runs目录下的模型文件
        runs_dir = os.path.join(self.current_dir, "runs", "detect")
        if os.path.exists(runs_dir):
            for subdir in os.listdir(runs_dir):
                subdir_path = os.path.join(runs_dir, subdir)
                if os.path.isdir(subdir_path):
                    weights_dir = os.path.join(subdir_path, "weights")
                    if os.path.exists(weights_dir):
                        for file in os.listdir(weights_dir):
                            if file.endswith('.pt'):
                                model_files.append(os.path.join(weights_dir, file))
        
        return model_files
    
    def prepare_training_data(self, info_widget):
        """
        准备训练数据
        """
        def prepare():
            try:
                # 获取数据源类型
                data_source = self.data_source_var.get()
                
                if data_source == "create":
                    info_widget.insert(tk.END, "\n正在创建示例训练数据...\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    
                    # 导入创建测试数据的模块
                    sys.path.append(self.current_dir)
                    from create_test_images import create_test_dataset
                    
                    # 创建训练数据
                    data_dir = os.path.join(self.current_dir, "fastfood_yolo_dataset")
                    create_test_dataset(data_dir)
                    
                    info_widget.insert(tk.END, f"示例训练数据已创建在: {data_dir}\n")
                    info_widget.insert(tk.END, "包含以下类别: Burger, Donut, Pizza\n")
                    info_widget.insert(tk.END, "每个类别包含20张训练图像和5张验证图像\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    
                else:
                    info_widget.insert(tk.END, "\n使用已有数据集进行训练\n")
                    info_widget.insert(tk.END, f"数据集路径: {self.existing_data_var.get()}\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                
                info_widget.insert(tk.END, "\n训练数据准备完成！\n")
                info_widget.insert(tk.END, "现在可以点击'开始训练'按钮开始训练模型\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
            except Exception as e:
                info_widget.insert(tk.END, f"\n准备训练数据时出错: {str(e)}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
        
        # 在新线程中执行数据准备
        threading.Thread(target=prepare, daemon=True).start()
    
    def start_training(self, info_widget):
        """
        开始训练模型
        """
        def train():
            try:
                # 获取训练参数
                data_dir = self.data_dir_var.get()
                model_name = self.model_var.get()
                epochs = int(self.epochs_var.get())
                img_size = int(self.img_size_var.get())
                batch_size = int(self.batch_size_var.get())
                
                info_widget.insert(tk.END, f"\n开始训练模型...\n")
                info_widget.insert(tk.END, f"数据目录: {data_dir}\n")
                info_widget.insert(tk.END, f"模型: {model_name}\n")
                info_widget.insert(tk.END, f"训练轮数: {epochs}\n")
                info_widget.insert(tk.END, f"图像尺寸: {img_size}\n")
                info_widget.insert(tk.END, f"批次大小: {batch_size}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                # 检查data.yaml文件是否存在
                data_yaml = os.path.join(data_dir, 'data.yaml')
                if not os.path.exists(data_yaml):
                    info_widget.insert(tk.END, f"错误: 数据配置文件 {data_yaml} 不存在!\n")
                    info_widget.insert(tk.END, "请先点击'准备训练数据'按钮创建训练数据。\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    return
                
                # 加载模型
                info_widget.insert(tk.END, f"加载预训练模型: {model_name}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                model = YOLO(model_name)
                
                # 开始训练
                info_widget.insert(tk.END, "开始训练，请耐心等待...\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                # 训练模型
                results = model.train(
                    data=data_yaml,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    project='runs/detect',
                    name='_classification',
                    exist_ok=True
                )
                
                info_widget.insert(tk.END, f"训练完成!\n")
                info_widget.insert(tk.END, f"模型保存在: runs/detect/fastfood_classification/weights/\n")
                info_widget.insert(tk.END, f"最佳模型: runs/detect/fastfood_classification/weights/best.pt\n")
                info_widget.insert(tk.END, f"最新模型: runs/detect/fastfood_classification/weights/last.pt\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
            except Exception as e:
                info_widget.insert(tk.END, f"\n训练过程中出错: {str(e)}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
        
        # 在新线程中执行训练
        threading.Thread(target=train, daemon=True).start()
    
    def test_image(self, info_widget, result_frame):
        """
        测试图像
        """
        def test():
            try:
                # 获取测试参数
                model_path = self.test_model_var.get()
                image_path = self.image_path_var.get()
                conf_threshold = float(self.conf_threshold_var.get())
                
                info_widget.insert(tk.END, f"\n开始测试图像...\n")
                info_widget.insert(tk.END, f"模型文件: {model_path}\n")
                info_widget.insert(tk.END, f"测试图像: {image_path}\n")
                info_widget.insert(tk.END, f"置信度阈值: {conf_threshold}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                # 检查模型文件是否存在
                if not os.path.exists(model_path):
                    info_widget.insert(tk.END, f"错误: 模型文件 {model_path} 不存在!\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    return
                
                # 检查图像文件是否存在
                if not os.path.exists(image_path):
                    info_widget.insert(tk.END, f"错误: 图像文件 {image_path} 不存在!\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    return
                
                # 加载模型
                info_widget.insert(tk.END, f"加载模型: {model_path}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                model = YOLO(model_path)
                
                # 读取图像
                info_widget.insert(tk.END, f"读取图像: {image_path}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                img = cv2.imread(image_path)
                if img is None:
                    info_widget.insert(tk.END, f"错误: 无法读取图像文件 {image_path}!\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                    return
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 进行预测
                info_widget.insert(tk.END, "正在进行目标检测...\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
                results = model(img_rgb, conf=conf_threshold)
                
                # 处理检测结果
                result_img = img.copy()
                detected_objects = []
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box.astype(int)
                        pred_class = model.names[class_id]
                        
                        # 绘制边界框
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 添加标签
                        label = f'{pred_class}: {conf:.2f}'
                        cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detected_objects.append(f"{pred_class}: {conf:.2f}")
                        
                        info_widget.insert(tk.END, f"检测到目标 {i+1}: {pred_class}, 置信度: {conf:.2f}\n")
                        info_widget.see(tk.END)
                        self.root.update_idletasks()
                else:
                    info_widget.insert(tk.END, "未检测到任何目标\n")
                    info_widget.see(tk.END)
                    self.root.update_idletasks()
                
                # 显示结果图像
                self.display_image(result_img, result_frame)
                
                # 显示检测摘要
                if detected_objects:
                    info_widget.insert(tk.END, f"\n检测完成，共检测到 {len(detected_objects)} 个目标:\n")
                    for obj in detected_objects:
                        info_widget.insert(tk.END, f"  - {obj}\n")
                else:
                    info_widget.insert(tk.END, "\n检测完成，未检测到任何目标\n")
                
                info_widget.see(tk.END)
                self.root.update_idletasks()
                
            except Exception as e:
                info_widget.insert(tk.END, f"\n测试过程中出错: {str(e)}\n")
                info_widget.see(tk.END)
                self.root.update_idletasks()
        
        # 在新线程中执行测试
        threading.Thread(target=test, daemon=True).start()
    
    def display_image(self, img, parent_frame):
        """
        在GUI中显示图像
        """
        # 调整图像大小以适应显示区域
        height, width = img.shape[:2]
        max_width = 400
        max_height = 400
        
        if width > max_width or height > max_height:
            # 计算缩放比例
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # 转换为PIL图像并显示
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_img)
        
        # 清除之前的图像
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # 显示新图像
        image_label = ttk.Label(parent_frame, image=photo)
        image_label.image = photo  # 保持引用
        image_label.pack(fill=tk.BOTH, expand=True)

def main():
    """
    主函数
    """
    root = tk.Tk()
    app = FastFoodClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()