import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "Golub_Leukemia": {
        "train_path": os.path.join(BASE_DIR, 'dataset', 'golub', 'data_set_ALL_AML_train.csv'),
        "test_path": os.path.join(BASE_DIR, 'dataset', 'golub', 'data_set_ALL_AML_independent.csv'),
        "label_col": None,
        "is_split": True 
    },
    "CuMiDa_GSE9476": {
        "path": os.path.join(BASE_DIR, 'dataset', 'cumida', 'Leukemia_GSE9476.csv'),
        "label_col": "type",
        "is_split": False 
    }
}

# --- ALGORITHM IDENTIFIERS ---
ALGO_NAME = "RBMO-SBOA"  #

# --- HYPERPARAMETERS ---
POP_SIZE = 20             
MAX_ITER = 100             
EXCHANGE_INTERVAL = 10     
PRINT_PROGRESS = True    
THRESHOLD = 0.5
K_MAX = 120                
R_RUNS = 10                # Số lần chạy chuẩn khoa học để tính Mean/STD
PATIENCE = 10              

# --- MULTI-OBJECTIVE WEIGHTS ---
ALPHA_ERROR = 0.7          # Trọng số sai số phân loại
BETA_REDUCTION = 0.2       # Trọng số giảm số lượng gen
GAMMA_STABILITY = 0.1      # Trọng số độ ổn định tập gen

# --- CLASSIFIER CONFIGURATION ---
RF_PARAMS = {'n_estimators': 50, 'n_jobs': 1, 'random_state': 42} # Cố định hạt giống ngẫu nhiên
N_FOLDS = 3
TEST_SIZE = 0.4