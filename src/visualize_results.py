import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import config
import matplotlib

# Cấu hình đồ họa không hiển thị cửa sổ (Agg) để tránh lỗi bộ nhớ
matplotlib.use('Agg') 
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# Định nghĩa màu sắc chuẩn
COLORS = {config.ALGO_NAME: '#D62728', 'GA': '#1F77B4', 'PSO': '#2CA02C'}
STYLES = {config.ALGO_NAME: '-', 'GA': '--', 'PSO': '-.'}
ORDER = [config.ALGO_NAME, 'GA', 'PSO']

def load_summary_data(dataset_name):
    """Đọc dữ liệu Mean/Std từ file report"""
    data_summary = []
    res_dir = os.path.join(config.BASE_DIR, "results", dataset_name)
    
    for algo in ORDER:
        file_path = os.path.join(res_dir, f"{algo}_parallel_report.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            try:
                row_mean = df[df['Run'] == 'Mean'].iloc[0]
                row_std = df[df['Run'] == 'Std'].iloc[0]
                data_summary.append({
                    'Algorithm': algo,
                    'Mean_Accuracy': float(row_mean['Accuracy']),
                    'Std_Accuracy': float(row_std['Accuracy']),
                    'Mean_Fitness': float(row_mean['Fitness']),
                    'Std_Fitness': float(row_std['Fitness']),
                })
            except IndexError:
                print(f"Warning: Malformed report file for {algo}")
                
    return pd.DataFrame(data_summary)

def draw_convergence_line(dataset_name, save_dir):
    """1. Vẽ biểu đồ ĐƯỜNG HỘI TỤ (Fig 1)"""
    plt.figure(figsize=(8, 5))
    has_data = False
    
    for algo in ORDER:
        hist_path = os.path.join(save_dir, f"{algo}_convergence.csv")
        if os.path.exists(hist_path):
            df_hist = pd.read_csv(hist_path)
            plt.plot(df_hist['Iteration'], df_hist['Mean_Fitness'], 
                     label=algo, color=COLORS[algo], linestyle=STYLES[algo], linewidth=2)
            has_data = True
            
    if has_data:
        plt.title(f'Average Convergence Analysis - {dataset_name}', fontweight='bold')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value (Fitness)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_Fig1_Convergence.png'), dpi=300)
        plt.close()
        print("   -> Generated: Convergence Line Chart")

def draw_accuracy_bar(df_sum, dataset_name, save_dir):
    """2. Vẽ biểu đồ CỘT ĐỘ CHÍNH XÁC (Fig 2)"""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_sum['Algorithm'], df_sum['Mean_Accuracy'], yerr=df_sum['Std_Accuracy'], capsize=5, 
                   color=[COLORS[a] for a in df_sum['Algorithm']], alpha=0.85, width=0.6)
    
    plt.ylim(0, 1.15) 
    plt.ylabel('Mean Accuracy')
    plt.title(f'Classification Accuracy (Mean ± Std) - {dataset_name}', fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                 f'{height*100:.2f}%', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_Fig2_Accuracy_Bar.png'), dpi=300)
    plt.close()
    print("   -> Generated: Accuracy Bar Chart")

def draw_boxplot_accuracy(dataset_name, save_dir):
    """3. Vẽ biểu đồ HỘP PHÂN BỐ (Fig 3)"""
    data_raw = []
    for algo in ORDER:
        path = os.path.join(save_dir, f"{algo}_parallel_report.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Lọc bỏ dòng Mean/Std để lấy dữ liệu thô
            df = df[~df['Run'].isin(['Mean', 'Std'])].copy()
            df['Algorithm'] = algo
            df['Accuracy'] = pd.to_numeric(df['Accuracy'])
            data_raw.append(df)
            
    if data_raw:
        df_all = pd.concat(data_raw)
        plt.figure(figsize=(8, 6))
        
        sns.boxplot(x='Algorithm', y='Accuracy', data=df_all, palette=COLORS, order=ORDER, width=0.5, linewidth=1.5)
        sns.stripplot(x='Algorithm', y='Accuracy', data=df_all, color='black', alpha=0.3, jitter=True, order=ORDER)
        
        plt.title(f'Accuracy Distribution - {dataset_name}', fontweight='bold')
        plt.ylabel('Accuracy Score')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{dataset_name}_Fig3_Accuracy_Boxplot.png'), dpi=300)
        plt.close()
        print("   -> Generated: Accuracy Boxplot")

def draw_fitness_bar(df_sum, dataset_name, save_dir):
    """4. Vẽ biểu đồ CỘT FITNESS (Fig 4) - Mới"""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_sum['Algorithm'], df_sum['Mean_Fitness'], yerr=df_sum['Std_Fitness'], capsize=5, 
                   color=[COLORS[a] for a in df_sum['Algorithm']], alpha=0.85, width=0.6)
    
    plt.ylabel('Mean Fitness Value (Lower is Better)')
    plt.title(f'Optimization Stability (Fitness) - {dataset_name}', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.001, 
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_Fig4_Fitness_Bar.png'), dpi=300)
    plt.close()
    print("   -> Generated: Fitness Bar Chart")

def draw_confusion_matrices(dataset_name, save_dir):
    """5. Vẽ MA TRẬN NHẦM LẪN (Fig 5) - Mới"""
    class_path = os.path.join(save_dir, "class_names.npy")
    if not os.path.exists(class_path): return
    
    class_names = np.load(class_path, allow_pickle=True)
    
    # Tìm các thuật toán có file CM
    algos_with_cm = [a for a in ORDER if os.path.exists(os.path.join(save_dir, f"{a}_best_cm.npy"))]
    n = len(algos_with_cm)
    if n == 0: return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1: axes = [axes]
    
    for i, algo in enumerate(algos_with_cm):
        cm = np.load(os.path.join(save_dir, f"{algo}_best_cm.npy"))
        ax = axes[i]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax, 
                    annot_kws={"size": 12, "weight": "bold"})
        
        ax.set_title(f"{algo} (Best Run)", fontweight='bold')
        ax.set_xlabel('Predicted Class')
        if i == 0: ax.set_ylabel('Actual Class')
        else: ax.set_ylabel('')

    plt.suptitle(f"Confusion Matrix Comparison - {dataset_name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{dataset_name}_Fig5_Confusion_Matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   -> Generated: Confusion Matrix Heatmap")

def main():
    print("VISUALIZATION STARTED...")
    datasets = ["Golub", "CuMiDa"]
    
    for ds in datasets:
        print(f"\nProcessing charts for: {ds}")
        save_dir = os.path.join(config.BASE_DIR, "results", ds)
        
        if os.path.exists(save_dir):
            # 1. Vẽ Line Chart (Convergence)
            draw_convergence_line(ds, save_dir)
            
            # 2. Vẽ Boxplot (Phân bố Accuracy)
            draw_boxplot_accuracy(ds, save_dir)
            
            # 3. Vẽ Confusion Matrix (So sánh lớp)
            draw_confusion_matrices(ds, save_dir)
            
            # 4. Vẽ Bar Charts (Mean/Std cho Acc & Fit)
            df_sum = load_summary_data(ds)
            if not df_sum.empty:
                draw_accuracy_bar(df_sum, ds, save_dir)
                draw_fitness_bar(df_sum, ds, save_dir)
        else:
            print(f"Error: Directory {save_dir} not found. Run parallel_main.py first.")
            
    print("\nALL CHARTS GENERATED SUCCESSFULLY.")

if __name__ == "__main__":
    main()