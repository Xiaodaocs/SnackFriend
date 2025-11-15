#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO快餐识别系统 - 增强版GUI程序
添加了营养信息显示功能
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import glob

# 导入营养数据处理模块
from nutrition_data_processor import NutritionDataProcessor

class FastFoodClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO快餐识别系统 - 增强版")
        self.root.geometry("1200x800")
        
        # 添加当前目录属性
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化营养数据处理器
        self.nutrition_processor = NutritionDataProcessor()
        
        # 类别名称映射 - 与训练模型匹配
        self.class_names = {
            0: 'Baked Potato',
            1: 'Burger',
            2: 'Crispy Chicken',
            3: 'Donut',
            4: 'Fries',
            5: 'Hot Dog',
            6: 'Pizza',
            7: 'Sandwich',
            8: 'Taco',
            9: 'Taquito'
        }
        
        # 创建主界面
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 创建选项卡
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建训练选项卡
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="模型训练")
        self.create_training_tab()
        
        # 创建测试选项卡
        self.testing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.testing_frame, text="模型测试")
        self.create_testing_tab()
    
    def create_training_tab(self):
        """创建训练选项卡"""
        # 左侧控制面板
        left_frame = ttk.Frame(self.training_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 数据集配置
        ttk.Label(left_frame, text="数据集配置", font=("Arial", 12, "bold")).pack(pady=5)
        
        ttk.Label(left_frame, text="数据集路径:").pack(anchor=tk.W)
        self.dataset_path_var = tk.StringVar(value=r"datasets\utkarshsaxenadn\fast-food-classification-dataset\versions\2\Fast Food Classification V2\Train")
        ttk.Entry(left_frame, textvariable=self.dataset_path_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="配置文件:").pack(anchor=tk.W)
        self.config_file_var = tk.StringVar(value="fastfood_yolo_dataset/data.yaml")
        ttk.Entry(left_frame, textvariable=self.config_file_var, width=30).pack(fill=tk.X, pady=2)
        
        # 训练参数
        ttk.Label(left_frame, text="训练参数", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        
        ttk.Label(left_frame, text="模型大小:").pack(anchor=tk.W)
        self.model_size_var = tk.StringVar(value="models/yolov8m.pt")
        model_size_combo = ttk.Combobox(left_frame, textvariable=self.model_size_var, width=27)
        model_size_combo['values'] = ('models/yolov8n.pt', 'models/yolov8s.pt', 'models/yolov8m.pt', 'models/yolov8l.pt', 'models/yolov8x.pt')
        model_size_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="训练轮数:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(left_frame, textvariable=self.epochs_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="批次大小:").pack(anchor=tk.W)
        self.batch_size_var = tk.StringVar(value="16")
        ttk.Entry(left_frame, textvariable=self.batch_size_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Label(left_frame, text="图像大小:").pack(anchor=tk.W)
        self.img_size_var = tk.StringVar(value="640")
        ttk.Entry(left_frame, textvariable=self.img_size_var, width=30).pack(fill=tk.X, pady=2)
        
        # 训练按钮
        self.train_button = ttk.Button(left_frame, text="开始训练", command=self.start_training)
        self.train_button.pack(pady=20, fill=tk.X)
        
        # 训练日志
        ttk.Label(left_frame, text="训练日志", font=("Arial", 12, "bold")).pack(pady=5)
        
        log_frame = ttk.Frame(left_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=15, width=40, yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # 右侧图表显示区域
        right_frame = ttk.Frame(self.training_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, text="训练结果", font=("Arial", 12, "bold")).pack(pady=5)
        
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_testing_tab(self):
        """创建测试选项卡"""
        # 左侧控制面板
        left_frame = ttk.Frame(self.testing_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # 模型选择
        ttk.Label(left_frame, text="模型选择", font=("Arial", 12, "bold")).pack(pady=5)
        
        ttk.Label(left_frame, text="模型路径:").pack(anchor=tk.W)
        self.model_path_var = tk.StringVar(value="models\yolov8m.pt")
        ttk.Entry(left_frame, textvariable=self.model_path_var, width=30).pack(fill=tk.X, pady=2)
        
        ttk.Button(left_frame, text="浏览模型", command=self.browse_model).pack(fill=tk.X, pady=2)
        
        # 测试参数
        ttk.Label(left_frame, text="测试参数", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        
        ttk.Label(left_frame, text="置信度阈值:").pack(anchor=tk.W)
        self.conf_threshold_var = tk.DoubleVar(value=0.25)
        conf_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, variable=self.conf_threshold_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=2)
        self.conf_label = ttk.Label(left_frame, text="0.25")
        self.conf_label.pack(anchor=tk.W)
        conf_scale.config(command=lambda v: self.conf_label.config(text=f"{float(v):.2f}"))
        
        ttk.Label(left_frame, text="IoU阈值:").pack(anchor=tk.W)
        self.iou_threshold_var = tk.DoubleVar(value=0.7)
        iou_scale = ttk.Scale(left_frame, from_=0.1, to=1.0, variable=self.iou_threshold_var, orient=tk.HORIZONTAL)
        iou_scale.pack(fill=tk.X, pady=2)
        self.iou_label = ttk.Label(left_frame, text="0.7")
        self.iou_label.pack(anchor=tk.W)
        iou_scale.config(command=lambda v: self.iou_label.config(text=f"{float(v):.2f}"))
        
        # 图像选择
        ttk.Label(left_frame, text="图像选择", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        
        ttk.Button(left_frame, text="选择图像", command=self.select_image).pack(fill=tk.X, pady=2)
        
        self.image_path_var = tk.StringVar(value="")
        ttk.Entry(left_frame, textvariable=self.image_path_var, width=30).pack(fill=tk.X, pady=2)
        
        # 检测按钮
        self.detect_button = ttk.Button(left_frame, text="开始检测", command=self.detect_objects)
        self.detect_button.pack(pady=20, fill=tk.X)
        
        # 右侧显示区域 - 创建上下两部分
        right_container = ttk.Frame(self.testing_frame)
        right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部分：图像显示
        image_frame = ttk.LabelFrame(right_container, text="检测结果")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="请选择图像进行检测", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 下部分：营养信息显示
        nutrition_frame = ttk.LabelFrame(right_container, text="营养信息")
        nutrition_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建营养信息显示区域，使用滚动文本框
        nutrition_scroll = ttk.Scrollbar(nutrition_frame)
        nutrition_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.nutrition_text = tk.Text(nutrition_frame, height=10, width=50, yscrollcommand=nutrition_scroll.set, wrap=tk.WORD)
        self.nutrition_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        nutrition_scroll.config(command=self.nutrition_text.yview)
        
        # 添加营养信息统计标签
        self.nutrition_summary_label = ttk.Label(nutrition_frame, text="总营养信息将在检测后显示", font=("Arial", 10, "bold"))
        self.nutrition_summary_label.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    def browse_model(self):
        """浏览模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_path_var.set(file_path)
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """显示图像"""
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 调整图像大小以适应显示区域
            image_width, image_height = image.size
            max_width = 600
            max_height = 400
            
            if image_width > max_width or image_height > max_height:
                ratio = min(max_width / image_width, max_height / image_height)
                new_width = int(image_width * ratio)
                new_height = int(image_height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为PhotoImage并显示
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图像: {e}")
    
    def log_message(self, message):
        """添加日志消息"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_training(self):
        """开始训练模型"""
        # 获取训练参数
        dataset_path = self.dataset_path_var.get()
        config_file = self.config_file_var.get()
        model_size = self.model_size_var.get()
        epochs = int(self.epochs_var.get())
        batch_size = int(self.batch_size_var.get())
        img_size = int(self.img_size_var.get())
        
        # 验证参数
        if not os.path.exists(config_file):
            messagebox.showerror("错误", f"配置文件不存在: {config_file}")
            return
        
        # 禁用训练按钮
        self.train_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)
        
        # 在新线程中运行训练
        threading.Thread(
            target=self.train_model,
            args=(dataset_path, config_file, model_size, epochs, batch_size, img_size),
            daemon=True
        ).start()
    
    def train_model(self, dataset_path, config_file, model_size, epochs, batch_size, img_size):
        """训练模型"""
        try:
            self.log_message("开始训练模型...")
            
            # 加载模型
            self.log_message(f"加载模型: {model_size}")
            model = YOLO(model_size)
            
            # 开始训练
            self.log_message(f"开始训练，轮数: {epochs}，批次大小: {batch_size}")
            results = model.train(
                data=config_file,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project='runs/detect',
                name='train'
            )
            
            self.log_message("训练完成!")
            
            # 显示训练结果图表
            self.display_training_results()
            
        except Exception as e:
            self.log_message(f"训练失败: {e}")
            messagebox.showerror("错误", f"训练失败: {e}")
        finally:
            # 重新启用训练按钮
            self.train_button.config(state=tk.NORMAL)
    
    def display_training_results(self):
        """显示训练结果图表"""
        try:
            # 查找最新训练结果目录
            result_dirs = glob.glob('runs/detect/train*')
            if not result_dirs:
                self.log_message("未找到训练结果目录")
                return
            
            # 按修改时间排序，获取最新的
            result_dirs.sort(key=os.path.getmtime, reverse=True)
            latest_dir = result_dirs[0]
            
            # 查找结果图像
            results_png = os.path.join(latest_dir, 'results.png')
            confusion_matrix_png = os.path.join(latest_dir, 'confusion_matrix.png')
            
            # 清除之前的图表
            self.fig.clear()
            
            # 显示结果图像
            if os.path.exists(results_png):
                img = plt.imread(results_png)
                ax = self.fig.add_subplot(121)
                ax.imshow(img)
                ax.set_title('训练结果')
                ax.axis('off')
            
            # 显示混淆矩阵
            if os.path.exists(confusion_matrix_png):
                img = plt.imread(confusion_matrix_png)
                ax = self.fig.add_subplot(122)
                ax.imshow(img)
                ax.set_title('混淆矩阵')
                ax.axis('off')
            
            self.canvas.draw()
            self.log_message(f"训练结果图表已加载: {latest_dir}")
            
        except Exception as e:
            self.log_message(f"加载训练结果失败: {e}")
    
    def detect_objects(self):
        """检测图像中的对象"""
        # 获取参数
        model_path = self.model_path_var.get()
        image_path = self.image_path_var.get()
        conf_threshold = self.conf_threshold_var.get()
        iou_threshold = self.iou_threshold_var.get()
        
        # 验证参数
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"模型文件不存在: {model_path}")
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("错误", f"图像文件不存在: {image_path}")
            return
        
        # 禁用检测按钮
        self.detect_button.config(state=tk.DISABLED)
        
        # 在新线程中运行检测
        threading.Thread(
            target=self.test_image,
            args=(model_path, image_path, conf_threshold, iou_threshold),
            daemon=True
        ).start()
    
    def test_image(self, model_path, image_path, conf_threshold, iou_threshold):
        """测试图像"""
        try:
            # 加载模型
            model = YOLO(model_path)
            
            # 进行预测
            results = model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                save=True,
                project='runs/detect',
                name='test'
            )
            
            # 获取检测结果
            result = results[0]
            
            # 显示结果图像
            result_image_path = os.path.join('runs/detect/test', os.path.basename(image_path))
            if os.path.exists(result_image_path):
                self.display_image(result_image_path)
            
            # 处理检测结果并显示营养信息
            self.process_nutrition_info(result, model)
            
        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {e}")
        finally:
            # 重新启用检测按钮
            self.detect_button.config(state=tk.NORMAL)
    
    def process_nutrition_info(self, result, model):
        """处理检测结果并显示营养信息"""
        try:
            # 清空营养信息显示区域
            self.nutrition_text.delete(1.0, tk.END)
            
            # 获取检测到的对象
            boxes = result.boxes
            if len(boxes) == 0:
                self.nutrition_text.insert(tk.END, "未检测到快餐对象")
                self.nutrition_summary_label.config(text="未检测到快餐对象")
                return
            
            # 存储每个检测到的对象的营养信息
            nutrition_list = []
            
            # 处理每个检测到的对象
            for i, box in enumerate(boxes):
                # 获取类别ID和置信度
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 获取类别名称
                # 检查类别ID是否在有效范围内
                if 0 <= class_id < len(self.class_names):
                    class_name = self.class_names.get(class_id, f"类别{class_id}")
                else:
                    # 如果类别ID超出范围，尝试从模型中获取类别名称
                    try:
                        class_name = model.names[class_id]
                    except:
                        class_name = f"未知类别({class_id})"
                
                # 获取营养信息
                nutrition_info = self.nutrition_processor.get_nutrition_info_by_category(class_name)
                
                if nutrition_info:
                    # 添加到营养信息列表
                    nutrition_list.append(nutrition_info)
                    
                    # 显示单个对象的营养信息
                    self.nutrition_text.insert(tk.END, f"检测对象 {i+1}: {class_name} (置信度: {confidence:.2f})\n")
                    self.nutrition_text.insert(tk.END, f"{self.nutrition_processor.format_nutrition_text(nutrition_info)}\n")
                    self.nutrition_text.insert(tk.END, "\n" + "="*50 + "\n\n")
                else:
                    self.nutrition_text.insert(tk.END, f"检测对象 {i+1}: {class_name} (置信度: {confidence:.2f})\n")
                    self.nutrition_text.insert(tk.END, "未找到相关营养信息\n\n")
            
            # 计算并显示总营养信息
            if nutrition_list:
                total_nutrition = self.nutrition_processor.calculate_total_nutrition(nutrition_list)
                if total_nutrition:
                    summary_text = f"总营养信息 (共{len(nutrition_list)}个快餐对象):\n"
                    summary_text += f"总卡路里: {total_nutrition['calories']} kcal\n"
                    summary_text += f"总脂肪: {total_nutrition['total_fat']} g\n"
                    summary_text += f"总蛋白质: {total_nutrition['protein']} g\n"
                    summary_text += f"总碳水化合物: {total_nutrition['total_carb']} g"
                    
                    self.nutrition_summary_label.config(text=summary_text)
                else:
                    self.nutrition_summary_label.config(text="计算总营养信息失败")
            else:
                self.nutrition_summary_label.config(text="未找到任何营养信息")
                
        except Exception as e:
            self.nutrition_text.insert(tk.END, f"处理营养信息时出错: {e}")
            self.nutrition_summary_label.config(text="处理营养信息时出错")

def main():
    """主函数"""
    root = tk.Tk()
    app = FastFoodClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()