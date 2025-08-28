import os
import re

def interactive_rename(directory):
    """
    交互式批量重命名工具
    """
    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    
    if not files:
        print("目录中没有文件")
        return
    
    # 显示当前文件列表
    print("当前文件列表:")
    for i, filename in enumerate(files, 1):
        print(f"{i:2d}. {filename}")
    
    # 选择操作
    print("\n选择操作:")
    print("1. 添加前缀")
    print("2. 添加后缀")
    print("3. 替换部分文本")
    print("4. 使用正则表达式替换")
    print("5. 添加序列号")
    
    choice = input("请输入选项 (1-5): ")
    
    if choice == "1":
        prefix = input("请输入要添加的前缀: ")
        for filename in files:
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, prefix + filename)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {prefix + filename}")
    
    elif choice == "2":
        suffix = input("请输入要添加的后缀: ")
        for filename in files:
            name, ext = os.path.splitext(filename)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, name + suffix + ext)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {name + suffix + ext}")
    
    elif choice == "3":
        old_text = input("请输入要替换的文本: ")
        new_text = input("请输入替换后的文本: ")
        for filename in files:
            new_filename = filename.replace(old_text, new_text)
            if new_filename != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
    
    elif choice == "4":
        pattern = input("请输入正则表达式模式: ")
        replacement = input("请输入替换内容: ")
        for filename in files:
            new_filename = re.sub(pattern, replacement, filename)
            if new_filename != filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
    
    elif choice == "5":
        digits = int(input("请输入序列号位数 (如 3 表示 001): "))
        start = int(input("请输入起始编号: "))
        for i, filename in enumerate(files, start=start):
            seq = str(i).zfill(digits)
            name, ext = os.path.splitext(filename)
            new_filename = f"{seq}_{name}{ext}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_filename}")
    
    else:
        print("无效选项")

# 使用示例
directory = "C:\\Users\\lizhenwang\\Desktop\\test_perturbation_data"
interactive_rename(directory)