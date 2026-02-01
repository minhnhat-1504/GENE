import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Bỏ StandardScaler ở đây để tránh data leakage
label_encoder = LabelEncoder()

def load_dataset(path, label_col=None):
    df = pd.read_csv(path)
    
    # 1. Loại bỏ các cột rác không cần thiết
    trash_cols = [c for c in df.columns if c.lower() in ['samples', 'sample', 'id', 'index'] 
                  or 'call' in c.lower() or 'unnamed' in c.lower()]
    df = df.drop(columns=trash_cols)

    # 2. Xử lý dữ liệu dựa trên cấu trúc file
    if 'Gene Description' in df.columns or 'Gene Accession Number' in df.columns:
        # Dạng Golub: Gene là hàng, Mẫu là cột
        gene_names = df.iloc[:, 1].values 
        X = df.iloc[:, 2:].values.T 
        
        # Gán nhãn thủ công cho Golub
        if X.shape[0] == 38: 
            y_raw = np.array(['ALL']*27 + ['AML']*11)
        elif X.shape[0] == 34: 
            y_raw = np.array(['ALL']*20 + ['AML']*14)
        else:
            y_raw = np.array(['Unknown']*X.shape[0])
    else:
        # Dạng CuMiDa: Gene là cột, Mẫu là hàng
        if label_col and label_col in df.columns:
            y_raw = df[label_col].values
            X_df = df.drop(columns=[label_col])
            X = X_df.values
            gene_names = X_df.columns.values
        else:
            X = df.iloc[:, :-1].values
            y_raw = df.iloc[:, -1].values
            gene_names = df.columns[:-1].values

    # 3. Chỉ Encode nhãn, KHÔNG chuẩn hóa X ở đây
    y = label_encoder.fit_transform(y_raw)
    
    # Trả về dữ liệu thô (raw data)
    return X.astype(float), y, gene_names