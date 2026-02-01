import time
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config, data_loader, fitness_wrapper, pa5_runner, fs_mapping

# --- HÀM TÍNH ĐỘ ỔN ĐỊNH ---
def calculate_stability_post_hoc(mask_list):
    if len(mask_list) < 2: return 0.0
    jaccard_sums = 0
    for i in range(len(mask_list) - 1):
        set_a = set(np.where(mask_list[i] == 1)[0])
        set_b = set(np.where(mask_list[i+1] == 1)[0])
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        jaccard_sums += intersection / union if union > 0 else 0
    return jaccard_sums / (len(mask_list) - 1)

# --- HÀM CHẠY ĐƠN LẺ ---
def execute_single_run(run_id, algo_type, X_train, y_train, X_test, y_test):
    np.random.seed(int(time.time()) + run_id * 1000)
    start_t = time.time()
    
    # 1. CHỌN THUẬT TOÁN (ĐÃ THAY ĐỔI ĐỂ SO SÁNH COMPONENT)
    if algo_type == config.ALGO_NAME: # RBMO-SBOA
        curr_z, curr_fit, hist = pa5_runner.run_rbmo_sboa(
            X_train, y_train, fitness_wrapper.fitness_feature_selection, reference_masks=None
        )
    elif algo_type == "RBMO": # Component 1
        curr_z, curr_fit, hist = pa5_runner.run_rbmo_standalone(
            X_train, y_train, fitness_wrapper.fitness_feature_selection
        )
    elif algo_type == "SBOA": # Component 2
        curr_z, curr_fit, hist = pa5_runner.run_sboa_standalone(
            X_train, y_train, fitness_wrapper.fitness_feature_selection
        )
    else:
        raise ValueError(f"Unknown algo: {algo_type}")

    # 2. Đánh giá kết quả
    m = fs_mapping.binarize_solution(curr_z, config.THRESHOLD, k_max=config.K_MAX)
    sel_idx = np.where(m == 1)[0]
    
    cm = None
    acc, sens, spec = 0, 0, 0
    
    if len(sel_idx) > 0:
        rf = RandomForestClassifier(n_estimators=config.RF_PARAMS['n_estimators'], n_jobs=1, random_state=42)
        rf.fit(X_train[:, sel_idx], y_train)
        y_pred = rf.predict(X_test[:, sel_idx])
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred, average='macro', zero_division=0)
        spec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Run': run_id + 1, 'Accuracy': acc, 'Fitness': curr_fit, 'Selected_Genes': len(sel_idx),
        'Sensitivity': sens, 'Specificity': spec, 'Time': time.time() - start_t,
        'Mask': m, 'History': hist, 'CM': cm
    }

# --- MAIN PROCESS ---
def main():
    MAX_WORKERS = 10 
    print(f"COMPONENT COMPARISON STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')} | Workers: {MAX_WORKERS}")

    datasets = [("Golub", config.DATASETS["Golub_Leukemia"]), ("CuMiDa", config.DATASETS["CuMiDa_GSE9476"])]
    
    # DANH SÁCH SO SÁNH MỚI: LAI vs THÀNH PHẦN
    algos = [config.ALGO_NAME, "RBMO", "SBOA"]

    for data_name, data_cfg in datasets:
        print(f"\n{'='*50}\nDATASET: {data_name}\n{'='*50}")
        res_dir = os.path.join(config.BASE_DIR, "results_components", data_name) # Lưu folder khác để không đè
        os.makedirs(res_dir, exist_ok=True)

        # Load & Preprocess
        if data_cfg.get('is_split'):
            X_train_raw, y_train, _ = data_loader.load_dataset(data_cfg['train_path'])
            X_test_raw, y_test, _ = data_loader.load_dataset(data_cfg['test_path'])
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
        else:
            X_all, y_all, _ = data_loader.load_dataset(data_cfg['path'], data_cfg['label_col'])
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_all, y_all, test_size=config.TEST_SIZE, stratify=y_all, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
        
        class_names = data_loader.label_encoder.classes_
        np.save(os.path.join(res_dir, "class_names.npy"), class_names)

        for algo in algos:
            print(f">>> Processing {algo}...")
            results_list = []
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(execute_single_run, r, algo, X_train, y_train, X_test, y_test): r for r in range(config.R_RUNS)}
                for future in as_completed(futures):
                    res = future.result()
                    results_list.append(res)
                    sys.stdout.write(f"\r    [Done] Run {res['Run']}/{config.R_RUNS} | Acc: {res['Accuracy']*100:.2f}%")
                    sys.stdout.flush()
            print("")

            results_list.sort(key=lambda x: x['Run'])
            
            # Save results (CM, Convergence, Report)
            best_run = min(results_list, key=lambda x: (-x['Accuracy'], x['Fitness']))
            if best_run['CM'] is not None:
                np.save(os.path.join(res_dir, f"{algo}_best_cm.npy"), best_run['CM'])

            all_hists = [r['History'] for r in results_list if r['History']]
            if all_hists:
                max_len = max(len(h) for h in all_hists)
                padded_hists = [h + [h[-1]] * (max_len - len(h)) for h in all_hists]
                mean_history = np.mean(padded_hists, axis=0)
                pd.DataFrame({'Iteration': range(1, len(mean_history)+1), 'Mean_Fitness': mean_history})\
                  .to_csv(os.path.join(res_dir, f"{algo}_convergence.csv"), index=False)

            df_rows = [{k: v for k, v in r.items() if k not in ['Mask', 'History', 'CM']} for r in results_list]
            all_masks = [r['Mask'] for r in results_list]
            stability = calculate_stability_post_hoc(all_masks)
            
            df_report = pd.DataFrame(df_rows)
            df_save = df_report.copy()
            df_save.loc['Mean'] = df_report.mean()
            df_save.loc['Std'] = df_report.std()
            df_save.loc['Mean', 'Run'] = 'Mean'; df_save.loc['Std', 'Run'] = 'Std'
            df_save.loc['Mean', 'Stability'] = stability
            
            df_save.to_csv(os.path.join(res_dir, f"{algo}_parallel_report.csv"), index=False)
            print(f"    -> Saved Report for {algo}")

    print("\nCOMPONENT COMPARISON COMPLETED.")

if __name__ == "__main__":
    main()