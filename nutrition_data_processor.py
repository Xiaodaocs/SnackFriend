#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快餐营养数据处理模块
用于加载和处理快餐营养信息
"""

import os
import pandas as pd
import numpy as np

class NutritionDataProcessor:
    """
    快餐营养数据处理器
    """
    
    def __init__(self, csv_path=None):
        """
        初始化营养数据处理器
        
        参数:
            csv_path (str): 营养数据CSV文件路径
        """
        self.nutrition_data = None
        self.csv_path = csv_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                 "datasets", "ulrikthygepedersen", 
                                                 "fastfood-nutrition", "versions", "1", 
                                                 "fastfood.csv")
        self.load_nutrition_data()
    
    def load_nutrition_data(self):
        """
        加载营养数据
        """
        try:
            self.nutrition_data = pd.read_csv(self.csv_path)
            print(f"成功加载营养数据，共 {len(self.nutrition_data)} 条记录")
            return True
        except Exception as e:
            print(f"加载营养数据失败: {e}")
            return False
    
    def get_nutrition_info(self, food_name):
        """
        根据食物名称获取营养信息
        
        参数:
            food_name (str): 食物名称
            
        返回:
            dict: 营养信息字典，如果找不到则返回None
        """
        if self.nutrition_data is None:
            return None
        
        # 尝试精确匹配
        matches = self.nutrition_data[self.nutrition_data['item'].str.lower() == food_name.lower()]
        
        if len(matches) == 0:
            # 尝试模糊匹配
            matches = self.nutrition_data[self.nutrition_data['item'].str.lower().str.contains(food_name.lower(), na=False)]
        
        if len(matches) > 0:
            # 返回第一个匹配项
            row = matches.iloc[0]
            return {
                'restaurant': row['restaurant'],
                'item': row['item'],
                'calories': row['calories'],
                'cal_fat': row['cal_fat'],
                'total_fat': row['total_fat'],
                'sat_fat': row['sat_fat'],
                'trans_fat': row['trans_fat'],
                'cholesterol': row['cholesterol'],
                'sodium': row['sodium'],
                'total_carb': row['total_carb'],
                'fiber': row['fiber'],
                'sugar': row['sugar'],
                'protein': row['protein'],
                'vit_a': row['vit_a'],
                'vit_c': row['vit_c'],
                'calcium': row['calcium']
            }
        
        return None
    
    def get_nutrition_info_by_category(self, category_name):
        """
        根据类别名称获取平均营养信息
        
        参数:
            category_name (str): 类别名称 (如 Burger, Donut, Pizza等)
            
        返回:
            dict: 平均营养信息字典，如果找不到则返回None
        """
        if self.nutrition_data is None:
            return None
        
        # 根据类别名称筛选相关食物
        category_keywords = {
            'Burger': ['burger', 'burger'],
            'Donut': ['donut', 'donut', 'doughnut'],
            'Pizza': ['pizza', 'pizza'],
            'Fries': ['fries', 'fries', 'french fries'],
            'Taco': ['taco', 'taco']
        }
        
        if category_name not in category_keywords:
            return None
        
        keywords = category_keywords[category_name]
        matches = pd.DataFrame()
        
        for keyword in keywords:
            keyword_matches = self.nutrition_data[self.nutrition_data['item'].str.lower().str.contains(keyword, na=False)]
            matches = pd.concat([matches, keyword_matches])
        
        if len(matches) > 0:
            # 计算平均值
            avg_nutrition = {
                'restaurant': 'Various',
                'item': f'Average {category_name}',
                'calories': round(matches['calories'].mean()),
                'cal_fat': round(matches['cal_fat'].mean()),
                'total_fat': round(matches['total_fat'].mean(), 1),
                'sat_fat': round(matches['sat_fat'].mean(), 1),
                'trans_fat': round(matches['trans_fat'].mean(), 1),
                'cholesterol': round(matches['cholesterol'].mean()),
                'sodium': round(matches['sodium'].mean()),
                'total_carb': round(matches['total_carb'].mean()),
                'fiber': round(matches['fiber'].mean(), 1),
                'sugar': round(matches['sugar'].mean(), 1),
                'protein': round(matches['protein'].mean()),
                'vit_a': round(matches['vit_a'].mean()),
                'vit_c': round(matches['vit_c'].mean()),
                'calcium': round(matches['calcium'].mean())
            }
            return avg_nutrition
        else:
            # 如果没有找到匹配项，提供默认营养信息
            default_nutrition = {
                'Burger': {
                    'restaurant': 'Various',
                    'item': 'Average Burger',
                    'calories': 610,
                    'cal_fat': 316,
                    'total_fat': 35.2,
                    'sat_fat': 13.7,
                    'trans_fat': 1.4,
                    'cholesterol': 98,
                    'sodium': 1065,
                    'total_carb': 41,
                    'fiber': 2.1,
                    'sugar': 8.6,
                    'protein': 31,
                    'vit_a': 12,
                    'vit_c': 8,
                    'calcium': 23
                },
                'Donut': {
                    'restaurant': 'Various',
                    'item': 'Average Donut',
                    'calories': 269,
                    'cal_fat': 135,
                    'total_fat': 15.0,
                    'sat_fat': 5.0,
                    'trans_fat': 0.5,
                    'cholesterol': 22,
                    'sodium': 220,
                    'total_carb': 31,
                    'fiber': 1.0,
                    'sugar': 15,
                    'protein': 4,
                    'vit_a': 2,
                    'vit_c': 0,
                    'calcium': 4
                },
                'Pizza': {
                    'restaurant': 'Various',
                    'item': 'Average Pizza',
                    'calories': 285,
                    'cal_fat': 117,
                    'total_fat': 13.0,
                    'sat_fat': 5.5,
                    'trans_fat': 0.3,
                    'cholesterol': 25,
                    'sodium': 640,
                    'total_carb': 30,
                    'fiber': 2.0,
                    'sugar': 4.0,
                    'protein': 12,
                    'vit_a': 8,
                    'vit_c': 5,
                    'calcium': 15
                },
                'Fries': {
                    'restaurant': 'Various',
                    'item': 'Average Fries',
                    'calories': 365,
                    'cal_fat': 225,
                    'total_fat': 25.0,
                    'sat_fat': 4.0,
                    'trans_fat': 0.2,
                    'cholesterol': 0,
                    'sodium': 250,
                    'total_carb': 48,
                    'fiber': 4.0,
                    'sugar': 0.3,
                    'protein': 4,
                    'vit_a': 0,
                    'vit_c': 15,
                    'calcium': 2
                },
                'Taco': {
                    'restaurant': 'Various',
                    'item': 'Average Taco',
                    'calories': 210,
                    'cal_fat': 99,
                    'total_fat': 11.0,
                    'sat_fat': 4.5,
                    'trans_fat': 0.5,
                    'cholesterol': 30,
                    'sodium': 460,
                    'total_carb': 20,
                    'fiber': 3.0,
                    'sugar': 1.0,
                    'protein': 10,
                    'vit_a': 2,
                    'vit_c': 0,
                    'calcium': 10
                }
            }
            
            if category_name in default_nutrition:
                return default_nutrition[category_name]
        
        return None
    
    def calculate_total_nutrition(self, nutrition_list):
        """
        计算多个食物的总营养信息
        
        参数:
            nutrition_list (list): 营养信息字典列表
            
        返回:
            dict: 总营养信息字典
        """
        if not nutrition_list:
            return None
        
        total_nutrition = {
            'restaurant': 'Multiple',
            'item': 'Total Nutrition',
            'calories': sum(item['calories'] for item in nutrition_list),
            'cal_fat': sum(item['cal_fat'] for item in nutrition_list),
            'total_fat': round(sum(item['total_fat'] for item in nutrition_list), 1),
            'sat_fat': round(sum(item['sat_fat'] for item in nutrition_list), 1),
            'trans_fat': round(sum(item['trans_fat'] for item in nutrition_list), 1),
            'cholesterol': sum(item['cholesterol'] for item in nutrition_list),
            'sodium': sum(item['sodium'] for item in nutrition_list),
            'total_carb': sum(item['total_carb'] for item in nutrition_list),
            'fiber': round(sum(item['fiber'] for item in nutrition_list), 1),
            'sugar': round(sum(item['sugar'] for item in nutrition_list), 1),
            'protein': sum(item['protein'] for item in nutrition_list),
            'vit_a': sum(item['vit_a'] for item in nutrition_list),
            'vit_c': sum(item['vit_c'] for item in nutrition_list),
            'calcium': sum(item['calcium'] for item in nutrition_list)
        }
        
        return total_nutrition
    
    def format_nutrition_text(self, nutrition_info):
        """
        格式化营养信息为可读文本
        
        参数:
            nutrition_info (dict): 营养信息字典
            
        返回:
            str: 格式化的营养信息文本
        """
        if not nutrition_info:
            return "未找到营养信息"
        
        text = f"食物: {nutrition_info['item']}\n"
        text += f"餐厅: {nutrition_info['restaurant']}\n\n"
        text += "营养成分:\n"
        text += f"卡路里: {nutrition_info['calories']} kcal\n"
        text += f"脂肪卡路里: {nutrition_info['cal_fat']} kcal\n"
        text += f"总脂肪: {nutrition_info['total_fat']} g\n"
        text += f"饱和脂肪: {nutrition_info['sat_fat']} g\n"
        text += f"反式脂肪: {nutrition_info['trans_fat']} g\n"
        text += f"胆固醇: {nutrition_info['cholesterol']} mg\n"
        text += f"钠: {nutrition_info['sodium']} mg\n"
        text += f"总碳水化合物: {nutrition_info['total_carb']} g\n"
        text += f"膳食纤维: {nutrition_info['fiber']} g\n"
        text += f"糖: {nutrition_info['sugar']} g\n"
        text += f"蛋白质: {nutrition_info['protein']} g\n"
        text += f"维生素A: {nutrition_info['vit_a']} %\n"
        text += f"维生素C: {nutrition_info['vit_c']} %\n"
        text += f"钙: {nutrition_info['calcium']} %"
        
        return text

# 测试代码
if __name__ == "__main__":
    # 创建营养数据处理器
    processor = NutritionDataProcessor()
    
    # 测试获取汉堡的营养信息
    burger_nutrition = processor.get_nutrition_info_by_category("Burger")
    if burger_nutrition:
        print("汉堡平均营养信息:")
        print(processor.format_nutrition_text(burger_nutrition))
    
    # 测试获取甜甜圈的营养信息
    donut_nutrition = processor.get_nutrition_info_by_category("Donut")
    if donut_nutrition:
        print("\n甜甜圈平均营养信息:")
        print(processor.format_nutrition_text(donut_nutrition))
    
    # 测试计算总营养信息
    if burger_nutrition and donut_nutrition:
        total_nutrition = processor.calculate_total_nutrition([burger_nutrition, donut_nutrition])
        if total_nutrition:
            print("\n汉堡和甜甜圈总营养信息:")
            print(processor.format_nutrition_text(total_nutrition))
    else:
        print("\n无法计算总营养信息，因为部分营养信息获取失败")