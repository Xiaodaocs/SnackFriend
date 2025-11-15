"""
检查数据集中包含的所有食物类别
"""

import os

def check_food_categories():
    """检查数据集中包含的所有食物类别"""
    # 数据集路径
    dataset_path = r"datasets\utkarshsaxenadn\fast-food-classification-dataset\versions\2\Fast Food Classification V2\Train"
    
    # 获取所有文件夹名称（即类别名称）
    if os.path.exists(dataset_path):
        categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        categories.sort()  # 按字母顺序排序
        
        print(f"数据集中包含的食物类别 (共{len(categories)}种):")
        for i, category in enumerate(categories):
            print(f"{i}: {category}")
        
        return categories
    else:
        print(f"数据集路径不存在: {dataset_path}")
        return []

if __name__ == "__main__":
    categories = check_food_categories()
    
    # 生成类别映射字典
    if categories:
        print("\n生成的类别映射字典:")
        print("class_names = {")
        for i, category in enumerate(categories):
            print(f"    {i}: '{category}',")
        print("}")
        
        # 保存到文件
        with open("food_categories.txt", "w", encoding="utf-8") as f:
            f.write(f"数据集中包含的食物类别 (共{len(categories)}种):\n")
            for i, category in enumerate(categories):
                f.write(f"{i}: {category}\n")
            
            f.write("\n生成的类别映射字典:\n")
            f.write("class_names = {\n")
            for i, category in enumerate(categories):
                f.write(f"    {i}: '{category}',\n")
            f.write("}\n")
        
        print("\n类别信息已保存到 food_categories.txt 文件")