import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib

# Thiết lập chế độ vẽ biểu đồ không giao diện (dành cho server/console)
matplotlib.use('Agg') 
plt.rcParams["font.family"] = "serif" # Font chuẩn cho các tạp chí khoa học quốc tế

# Import các module nội bộ
import config, data_loader, fitness_wrapper, pa5_runner, fs_mapping

# --- CLASS GHI LOG AN TOÀN (FIX LỖI I/O) ---
class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # Mở file log với mã hóa utf-8
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        # Luôn ghi ra màn hình terminal
        self.terminal.write(message)
        # Chỉ ghi vào file nếu file vẫn đang mở
        if self.log and not self.log.closed:
            try:
                self.log.write(message)
            except ValueError:
                pass # Tránh lỗi I/O khi hệ thống đóng file bất ngờ

    def flush(self):
        # Đảm bảo luồng dữ liệu được đẩy đi
        self.terminal.flush()
        if self.log and not self.log.closed:
            try:
                self.log.flush()
            except ValueError:
                pass
        
    def close(self):
        """Đóng file log một cách an toàn"""
        if self.log:
            try:
                self.log.flush()
                self.log.close()
            except ValueError:
                pass
            self.log = None

# --- CÁC HÀM HỖ TRỢ VẼ BIỂU ĐỒ CHUYÊN NGHIỆP ---

def plot_combined_charts(results_store, data_name, res_dir):
    """Vẽ biểu đồ hội tụ và so sánh hiệu suất chuẩn Academic (DPI 300)"""
    print(f"\n[PLOT] Generating professional charts for {data_name}...")
    
    # Định nghĩa màu sắc và kiểu đường vẽ cố định
    colors = {config.ALGO_NAME: '#D62728', 'GA': '#1F77B4', 'PSO': '#2CA02C'}
    styles = {config.ALGO_NAME: '-', 'GA': '--', 'PSO': '-.'}
    algos = list(results_store.keys())

    # --- 1. Convergence Characteristics ---
    plt.figure(figsize=(8, 5))
    has_history = False
    for algo, data in results_store.items():
        if algo != 'Baseline' and 'history' in data and data['history']:
            plt.plot(data['history'], label=algo, color=colors.get(algo, 'black'), 
                     linestyle=styles.get(algo, '-'), linewidth=1.8)
            has_history = True
            
    if has_history:
        plt.title(f'Convergence Characteristics: {data_name} Dataset', fontsize=12, fontweight='bold')
        plt.xlabel('Number of Iterations', fontsize=11)
        plt.ylabel('Objective Function Value (Fitness)', fontsize=11)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(res_dir, 'FIG_Convergence_Analysis.png'), dpi=300)
        plt.close()
    
    # --- 2. Accuracy Comparison (Mean ± STD) ---
    plt.figure(figsize=(8, 5))
    acc_means = []
    acc_stds = [] 
    plot_algos = []
    
    for a in algos:
        plot_algos.append(a)
        if 'stats_mean' in results_store[a]:
            acc_means.append(results_store[a]['stats_mean']['Accuracy'])
            acc_stds.append(results_store[a]['stats_std']['Accuracy'])
        else:
            acc_means.append(results_store[a]['acc']) # Baseline
            acc_stds.append(0)

    bar_colors = ['#7F7F7F' if a == 'Baseline' else colors.get(a, '#FF7F0E') for a in plot_algos]
    
    bars = plt.bar(plot_algos, acc_means, yerr=acc_stds, capsize=6, color=bar_colors, alpha=0.85, width=0.6)
    plt.ylim(0, 1.15)
    plt.title(f'Classification Performance Comparison ({config.R_RUNS} runs)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Accuracy (%)', fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.2f}%', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'FIG_Accuracy_Comparison.png'), dpi=300)
    plt.close()

def plot_combined_confusion_matrices(results_store, data_name, res_dir, class_names):
    """Vẽ gộp các Confusion Matrices trên cùng một hàng"""
    algos = [k for k in results_store.keys()]
    n = len(algos)
    if n == 0: return

    print(f"[PLOT] Generating Combined Confusion Matrices...")
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]

    for i, algo in enumerate(algos):
        cm = results_store[algo].get('cm')
        ax = axes[i]
        
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            acc_val = results_store[algo]['acc'] * 100
            ax.set_title(f"{algo}\n(Best Acc: {acc_val:.1f}%)", fontweight='bold')
            ax.set_xlabel('Predicted Class')
            if i == 0: ax.set_ylabel('Actual Class')
            else: ax.set_ylabel('')
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')

    plt.suptitle(f"Confusion Matrix Comparison - {data_name}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(res_dir, 'FIG_Combined_Confusion_Matrices.png'), dpi=300)
    plt.close()

# --- HÀM THỰC THI (CORE LOGIC) ---
def run_experiment(data_name, task_cfg, algo_type=config.ALGO_NAME):
    """Thực thi một vòng thí nghiệm cho thuật toán được chỉ định"""
    if 'k_max' in task_cfg: config.K_MAX = task_cfg['k_max']
    
    print(f"\n{'-'*65}")
    print(f"Algorithm: {algo_type} | Dataset: {data_name}")
    print(f"{'-'*65}")

    # 1. Tải dữ liệu
    if task_cfg.get('is_split'):
        X_train, y_train, gene_names = data_loader.load_dataset(task_cfg['train_path'])
        X_test, y_test, _ = data_loader.load_dataset(task_cfg['test_path'])
    else:
        X_all, y_all, gene_names = data_loader.load_dataset(task_cfg['path'], task_cfg['label_col'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=42
        )
    
    class_names = data_loader.label_encoder.classes_
    rf_params = config.RF_PARAMS.copy()

    # 2. Đánh giá Baseline (Sử dụng toàn bộ đặc trưng ban đầu)
    if algo_type == "Baseline":
        rf_full = RandomForestClassifier(**rf_params)
        rf_full.fit(X_train, y_train)
        y_pred_full = rf_full.predict(X_test)
        acc_full = accuracy_score(y_test, y_pred_full)
        print(f"    + Baseline Accuracy: {acc_full*100:.2f}%")
        return {
            "acc": acc_full, "history": [], "genes_count": X_train.shape[1], 
            "cm": confusion_matrix(y_test, y_pred_full)
        }, class_names

    # 3. Vòng lặp tối ưu hóa và đánh giá thống kê
    all_best_masks = []
    run_stats = [] 
    final_best_z, final_best_fit = None, np.inf
    final_history = []
    
    for r in range(config.R_RUNS):
        start_opt = time.time()
        
        # Gọi các thuật toán tối ưu từ pa5_runner
        if algo_type == config.ALGO_NAME:
            curr_z, curr_fit, hist = pa5_runner.run_rbmo_sboa(X_train, y_train, fitness_wrapper.fitness_feature_selection, reference_masks=all_best_masks)
        elif algo_type == "GA":
            curr_z, curr_fit, hist = pa5_runner.run_ga(X_train, y_train, fitness_wrapper.fitness_feature_selection, reference_masks=all_best_masks)
        elif algo_type == "PSO":
            curr_z, curr_fit, hist = pa5_runner.run_pso(X_train, y_train, fitness_wrapper.fitness_feature_selection, reference_masks=all_best_masks)
        
        duration = time.time() - start_opt
        
        # Ánh xạ nghiệm sang tập gen được chọn
        m = fs_mapping.binarize_solution(curr_z, config.THRESHOLD, k_max=config.K_MAX)
        all_best_masks.append(m)
        
        sel_idx_run = np.where(m == 1)[0]
        if len(sel_idx_run) > 0:
            rf_run = RandomForestClassifier(**rf_params)
            rf_run.fit(X_train[:, sel_idx_run], y_train)
            y_pred_run = rf_run.predict(X_test[:, sel_idx_run])
            acc_run = accuracy_score(y_test, y_pred_run)
            sens_run = recall_score(y_test, y_pred_run, average='macro', zero_division=0)
            spec_run = precision_score(y_test, y_pred_run, average='macro', zero_division=0)
        else:
            acc_run, sens_run, spec_run = 0, 0, 0

        run_stats.append({
            'Accuracy': acc_run, 'Sensitivity': sens_run, 'Specificity': spec_run,
            'Selected_Genes': len(sel_idx_run), 'Fitness': curr_fit, 'Time(s)': duration
        })

        if curr_fit < final_best_fit:
            final_best_fit, final_best_z, final_history = curr_fit, curr_z, hist
        
        print(f"    -> Run {r+1}/{config.R_RUNS} | Fit: {curr_fit:.4f} | Acc: {acc_run*100:.2f}%")

    # 4. Tính toán số liệu thống kê Mean/STD
    df_stats = pd.DataFrame(run_stats)
    stats_mean = df_stats.mean()
    stats_std = df_stats.std()
    
    # 5. Lấy kết quả tốt nhất để vẽ Confusion Matrix
    _, fit_details = fitness_wrapper.fitness_feature_selection(final_best_z, X_train, y_train, config)
    sel_idx = np.where(fit_details['mask'] == 1)[0]
    
    rf_opt = RandomForestClassifier(**rf_params)
    rf_opt.fit(X_train[:, sel_idx], y_train)
    y_pred = rf_opt.predict(X_test[:, sel_idx])
    
    # Lưu báo cáo thống kê cho từng thuật toán
    res_dir = os.path.join(config.BASE_DIR, "results", data_name)
    df_stats.to_csv(os.path.join(res_dir, f'{algo_type}_statistical_report.csv'), index=False)

    return {
        "acc": accuracy_score(y_test, y_pred),
        "stats_mean": stats_mean, 
        "stats_std": stats_std,
        "history": final_history,
        "genes_count": len(sel_idx),
        "cm": confusion_matrix(y_test, y_pred)
    }, class_names

def main():
    # Tạo thư mục kết quả
    log_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(log_dir, exist_ok=True)
    
    # Khởi tạo logger hệ thống
    logger = DualLogger(os.path.join(log_dir, "EXECUTION_LOG.txt"))
    sys.stdout = logger

    try:
        print(f"SESSION STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"SYSTEM CONFIG: Pop={config.POP_SIZE}, Iter={config.MAX_ITER}, Runs={config.R_RUNS}")

        target_datasets = [
            ("Golub", config.DATASETS["Golub_Leukemia"]),
            ("CuMiDa", config.DATASETS["CuMiDa_GSE9476"])
        ]
        
        algos_to_run = [config.ALGO_NAME, "GA", "PSO"] 

        for data_name, data_cfg in target_datasets:
            res_dir = os.path.join(config.BASE_DIR, "results", data_name)
            os.makedirs(res_dir, exist_ok=True)
            results_store = {}
            
            # Phase 1: Baseline Evaluation
            base_res, c_names = run_experiment(data_name, data_cfg, "Baseline")
            results_store['Baseline'] = base_res
            
            # Phase 2: Hybrid and Metaheuristic Optimization
            for algo in algos_to_run:
                res, _ = run_experiment(data_name, data_cfg, algo)
                results_store[algo] = res

            # Phase 3: Scientific Visualization
            plot_combined_charts(results_store, data_name, res_dir)
            plot_combined_confusion_matrices(results_store, data_name, res_dir, c_names)
            
        print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Quan trọng: Luôn đóng logger để giải phóng file
        logger.close()

if __name__ == "__main__":
    main()