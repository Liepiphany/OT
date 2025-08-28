import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CSVStandardizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV数据行标准化工具（无表头）")
        self.root.geometry("900x700")
        
        # 变量初始化
        self.original_df = None
        self.processed_df = None
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # 标题
        Label(main_frame, text="CSV数据行标准化工具（无表头）", font=("Arial", 16, "bold")).pack(pady=10)
        
        # 说明文本
        Label(main_frame, text="上传没有表头的CSV文件，程序将计算每行的平均值，并将每个数据除以该行的平均值", 
              wraplength=600, justify="left").pack(pady=5)
        
        # 按钮框架
        button_frame = Frame(main_frame)
        button_frame.pack(pady=10)
        
        Button(button_frame, text="上传CSV文件", command=self.load_csv, 
               bg="#4CAF50", fg="white", padx=10).pack(side=LEFT, padx=5)
        Button(button_frame, text="处理数据", command=self.process_data, 
               bg="#2196F3", fg="white", padx=10).pack(side=LEFT, padx=5)
        Button(button_frame, text="保存结果", command=self.save_result, 
               bg="#FF9800", fg="white", padx=10).pack(side=LEFT, padx=5)
        
        # 数据显示区域
        data_frame = Frame(main_frame)
        data_frame.pack(fill=BOTH, expand=True, pady=10)
        
        # 原始数据标签和文本框
        Label(data_frame, text="原始数据", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=W)
        self.original_text = Text(data_frame, height=12, width=50)
        self.original_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # 处理后的数据标签和文本框
        Label(data_frame, text="处理后的数据（除以行平均值）", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=W)
        self.processed_text = Text(data_frame, height=12, width=50)
        self.processed_text.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # 配置网格权重
        data_frame.grid_rowconfigure(1, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        data_frame.grid_columnconfigure(1, weight=1)
        
        # 添加滚动条
        scrollbar1 = Scrollbar(data_frame, command=self.original_text.yview)
        scrollbar1.grid(row=1, column=0, sticky=E+N+S, padx=(0, 5))
        self.original_text.config(yscrollcommand=scrollbar1.set)
        
        scrollbar2 = Scrollbar(data_frame, command=self.processed_text.yview)
        scrollbar2.grid(row=1, column=1, sticky=E+N+S, padx=(0, 5))
        self.processed_text.config(yscrollcommand=scrollbar2.set)
        
        # 图表区域
        Label(main_frame, text="数据可视化", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor=W)
        self.figure_frame = Frame(main_frame, bg="white", height=250)
        self.figure_frame.pack(fill=BOTH, expand=True, pady=5)
        
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # 读取没有表头的CSV文件
                self.original_df = pd.read_csv(file_path, header=None)
                self.display_data(self.original_df, self.original_text)
                messagebox.showinfo("成功", f"已加载CSV文件: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文件: {str(e)}")
    
    def process_data(self):
        if self.original_df is None:
            messagebox.showwarning("警告", "请先上传CSV文件")
            return
            
        try:
            # 确保所有数据都是数值型
            df_numeric = self.original_df.apply(pd.to_numeric, errors='coerce')
            
            # 计算每行的平均值
            row_means = df_numeric.mean(axis=1)
            
            # 处理可能出现的除零错误
            # 如果某行的平均值为0，则将该行所有值设置为0（或NaN，这里设为0）
            row_means[row_means == 0] = np.nan
            
            # 将每行数据除以该行的平均值
            self.processed_df = df_numeric.div(row_means, axis=0)
            
            # 将NaN值替换为0（处理除零情况）
            self.processed_df = self.processed_df.fillna(0)
            
            # 显示处理后的数据
            self.display_data(self.processed_df, self.processed_text)
            
            # 显示可视化
            self.show_visualization()
            
            messagebox.showinfo("成功", "数据处理完成！")
        except Exception as e:
            messagebox.showerror("错误", f"处理数据时出错: {str(e)}")
    
    def display_data(self, df, text_widget):
        text_widget.delete(1.0, END)
        text_widget.insert(END, df.to_string(header=False, index=False))
    
    def show_visualization(self):
        # 清除之前的图表
        for widget in self.figure_frame.winfo_children():
            widget.destroy()
            
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # 原始数据的热力图
        im1 = ax1.imshow(self.original_df.values, cmap='viridis', aspect='auto')
        ax1.set_title('原始数据热力图')
        fig.colorbar(im1, ax=ax1)
        
        # 处理后数据的热力图
        im2 = ax2.imshow(self.processed_df.values, cmap='viridis', aspect='auto')
        ax2.set_title('标准化后数据热力图')
        fig.colorbar(im2, ax=ax2)
        
        # 调整布局
        fig.tight_layout()
        
        # 在Tkinter中嵌入图表
        canvas = FigureCanvasTkAgg(fig, self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    def save_result(self):
        if self.processed_df is None:
            messagebox.showwarning("警告", "没有处理后的数据可保存")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # 保存没有表头的CSV文件
                self.processed_df.to_csv(file_path, header=False, index=False)
                messagebox.showinfo("成功", f"结果已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存文件时出错: {str(e)}")

if __name__ == "__main__":
    root = Tk()
    app = CSVStandardizer(root)
    root.mainloop()