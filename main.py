import numpy as np
import pandas as pd
import time
from MLDetectTool import load_and_scale_data, create_sequence, train_isolation_forest, apply_anomaly_detection, \
    save_top_anomalies_json


def main():
    total_start_time = time.time()

    path1 = "data/data_hl19_1.csv"
    path2 = "data/data_hl19_2.csv"
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    window_len = 6
    contamination = 0.05

    scaled_df1 = load_and_scale_data(path1, 1)
    scaled_df2 = load_and_scale_data(path2, 2)

    # Fill phần đầu nếu có NaN bằng nội suy
    scaled_df1 = scaled_df1.ffill()
    scaled_df2 = scaled_df2.ffill()

    # Reset index để về dạng bảng thông thường
    scaled_df1 = scaled_df1.reset_index(drop=True)
    scaled_df2 = scaled_df2.reset_index(drop=True)

    print(f"\nSố dòng dữ liệu gốc - Node 1: {len(df1)}, Node 2: {len(df2)}")
    print(f"Số dòng dữ liệu scaled - Node 1: {len(scaled_df1)}, Node 2: {len(scaled_df2)}")

    X1 = scaled_df1
    X2 = scaled_df2
    #X1 = create_sequence(scaled_df1, window_len)
    #X2 = create_sequence(scaled_df2, window_len)

    print(f"Số window - Node 1: {len(X1)}, Node 2: {len(X2)}\n")

    #X = np.vstack([X1, X2])
    #X_flat = X.reshape(X.shape[0], -1)
    #X_flat.drop(columns=['date'], inplace=True)

    X_flat = pd.concat([scaled_df1, scaled_df2], ignore_index=True)

    # Loại bỏ các cột không dùng cho mô hình
    cols_drop = [c for c in X_flat.columns if "date" in c.lower() or "window" in c.lower() or "node" in c.lower()]
    X_flat = X_flat.drop(columns=cols_drop, errors="ignore")

    # Train
    iso = train_isolation_forest(X_flat, contamination)

    # Dự đoán và ánh xạ về từng điểm gốc
    data_with_date = apply_anomaly_detection(
        iso, X_flat, X1,X2, scaled_df1, scaled_df2, window_len
    )
    #print(data_with_date.head())

    print(f"\nSố điểm dữ liệu sau khi tính trung bình window: {len(data_with_date)}")
    print(f"Số điểm bất thường phát hiện: {(data_with_date['anomaly'] == -1).sum()}\n")

    anomalies = save_top_anomalies_json(data_with_date, df1, df2, window_len, output_path="result_ml.json")

    total_elapsed_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"TỔNG THỜI GIAN THỰC THI: {total_elapsed_time:.2f} giây ({total_elapsed_time / 60:.2f} phút)")
    print(f"{'=' * 60}")

# Hàm main chỉ để sinh result_ml.json và thời gian xử lý
if __name__ == "__main__":
    main()