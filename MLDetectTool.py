import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_and_scale_data(path, node_id):
    start_time = time.time()

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df['node'] = node_id
    df = df.set_index('date')

    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time')

    percent_cols = [c for c in df.columns if 'UTIL' in c or c.endswith('SR') or c.endswith('EASR')]
    throughput_cols = [c for c in df.columns if 'THROUGHPUT' in c and c not in percent_cols]
    count_cols = [c for c in df.columns if ('SAU' in c or 'BEARER' in c or 'SESSION' in c)
                  and c not in percent_cols and c not in throughput_cols]
    error_cols = [c for c in df.columns if ('FAIL' in c or 'DROP' in c or 'ERROR' in c)
                  and c not in percent_cols and c not in throughput_cols and c not in count_cols]

    scaled_df = df.copy()

    # Scale theo từng nhóm đặc trưng
    if percent_cols:
        scaler_percent = MinMaxScaler()
        scaled_df[percent_cols] = scaler_percent.fit_transform(df[percent_cols])

    if throughput_cols:
        scaler_thr = StandardScaler()
        scaled_df[throughput_cols] = scaler_thr.fit_transform(df[throughput_cols])

    if count_cols:
        scaler_count = RobustScaler()
        scaled_df[count_cols] = scaler_count.fit_transform(df[count_cols])

    if error_cols:
        df[error_cols] = np.log1p(df[error_cols])
        scaler_err = MinMaxScaler()
        scaled_df[error_cols] = scaler_err.fit_transform(df[error_cols])

    scaled_df = scaled_df.ffill().reset_index()

    elapsed_time = time.time() - start_time
    print(f"[load_and_scale_data] Node {node_id} hoàn thành trong {elapsed_time:.2f} giây")

    return scaled_df


def create_sequence(df, window_len):
    start_time = time.time()

    df_ = df.drop(columns=['date'])
    n = len(df_)
    sequences = []

    for i in range(0, n, window_len):
        window = df_.iloc[i:i + window_len].values
        if len(window) < window_len:
            last_row = window[-1]
            padding = np.tile(last_row, (window_len - len(window), 1))
            window = np.vstack([window, padding])

        sequences.append(window)

    elapsed_time = time.time() - start_time
    print(f"[create_sequence] Tạo {len(sequences)} sequences trong {elapsed_time:.2f} giây")

    return np.array(sequences)


def train_isolation_forest(X_flat, contamination=0.05, random_state=42):
    start_time = time.time()

    X_train, X_temp = train_test_split(X_flat, test_size=0.3, random_state=random_state)
    X_valid, X_test = train_test_split(X_temp, test_size=0.5, random_state=random_state)

    print(f"[train_isolation_forest] Train size: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state
    )
    iso.fit(X_train)

    elapsed_time = time.time() - start_time
    print(f"[train_isolation_forest] Training hoàn thành trong {elapsed_time:.2f} giây")
    return iso


def apply_anomaly_detection(iso, X, X1, X2, scaled_df1, scaled_df2, window_len):

    start_time = time.time()

    # --- Dự đoán anomaly ---
    y_pred_full = iso.predict(X)          # -1 = anomaly, 1 = normal
    scores_full = iso.decision_function(X) * -1  # đảo dấu -> score cao = càng bất thường

    # --- Tách theo node ---
    n1 = len(X1)
    scores1, scores2 = scores_full[:n1], scores_full[n1:]
    y1, y2 = y_pred_full[:n1], y_pred_full[n1:]

    # --- Chỉ thêm thông tin window_start / window_end ---
    def add_window_info(df, window_len):
        df = df.copy()
        df["window_start"] = df["date"]
        df["window_end"] = df["date"] + pd.to_timedelta(window_len, unit="m")  # ví dụ: window_len phút
        return df

    df1_result = add_window_info(scaled_df1, window_len)
    df2_result = add_window_info(scaled_df2, window_len)

    # --- Thêm cột kết quả từ mô hình ---
    df1_result["score"] = scores1
    df1_result["anomaly"] = y1
    df2_result["score"] = scores2
    df2_result["anomaly"] = y2

    # --- Gộp lại ---
    data_with_date = pd.concat([df1_result, df2_result], ignore_index=True)

    elapsed_time = time.time() - start_time
    print(f"[apply_anomaly_detection] Hoàn thành trong {elapsed_time:.2f} giây")

    return data_with_date


#Lưu toàn bộ điểm bất thường (anomaly == -1) với giá trị trung bình của window.
def save_top_anomalies_json(data_with_date, df_raw1, df_raw2, window_len, output_path="result_ml.json"):
    """
    Lưu các điểm anomaly (không tính trung bình theo window nữa).
    Mỗi window chỉ dùng để xác định thời gian start / end.
    """
    import pandas as pd, time

    start_time = time.time()

    def attach_windows(df, window_len):
        """Tạo cột window_start và window_end cho từng đoạn."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        n = len(df)
        num_windows = (n + window_len - 1) // window_len
        window_starts, window_ends = [], []

        for i in range(num_windows):
            start_idx = i * window_len
            end_idx = min(start_idx + window_len, n)
            ws = df.loc[start_idx, "date"]
            we = df.loc[end_idx - 1, "date"]
            window_starts.extend([ws] * (end_idx - start_idx))
            window_ends.extend([we] * (end_idx - start_idx))

        df["window_start"] = window_starts
        df["window_end"] = window_ends
        return df

    # --- Gắn window cho từng node ---
    df_raw1 = attach_windows(df_raw1, window_len)
    df_raw1["node"] = 1

    df_raw2 = attach_windows(df_raw2, window_len)
    df_raw2["node"] = 2

    # --- Gộp dữ liệu 2 node ---
    df_all = pd.concat([df_raw1, df_raw2], ignore_index=True)

    # --- Thêm score và anomaly từ mô hình ---
    df_all["score"] = data_with_date["score"].values
    df_all["anomaly"] = data_with_date["anomaly"].values

    # --- Lọc anomaly ---
    anomalies = df_all[df_all["anomaly"] == -1].copy()
    anomalies = anomalies.sort_values("score", ascending=False)

    # --- Lưu file ---
    #full_path = output_path.replace(".json", "_full.json")
    #df_all.to_json(full_path, orient="records", date_format="iso", force_ascii=False)
    anomalies.to_json(output_path, orient="records", date_format="iso", force_ascii=False)

    elapsed = time.time() - start_time
    print(f"[save_top_anomalies_json] ✅ Lưu {len(anomalies)} anomaly")
    print(f"   → File anomaly: {output_path}")
    print(f"   → Thời gian: {elapsed:.2f} giây")

    return anomalies



def extract_anomaly_info(json_path):
    import json
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"File {json_path} rỗng.")
                return pd.DataFrame()
            data = json.loads(content)
    except Exception as e:
        print(f"Không thể đọc JSON {json_path}: {e}")
        return pd.DataFrame()

    records = []
    for anomaly_obj in data:
        try:
            # anomaly_obj = {"anomaly_1": {...}}
            info = list(anomaly_obj.values())[0]
            meta = info.get("meta", {})
            name_fields = info.get("name_field", [])
            date = meta.get("window_start") or meta.get("date")
            node = meta.get("node")

            if not date or not name_fields:
                continue

            # Tạo bản ghi cho từng field
            for f_name in name_fields:
                records.append({
                    "date": pd.to_datetime(date, errors="coerce"),
                    "node": int(node) if node is not None else None,
                    "field": f_name.strip()
                })

        except Exception as e:
            print(f"Lỗi khi xử lý anomaly_obj: {e}")
            continue

    df = pd.DataFrame(records)
    if df.empty:
        print(f"Không trích xuất được anomaly nào từ {json_path}.")
    else:
        print(f"Đã trích xuất {len(df)} anomaly records từ {json_path}.")
        print(df)
    return df



def get_window_averages(df, window_len):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    result_rows = []
    n = len(df)
    num_windows = int(np.ceil(n / window_len))

    for i in range(num_windows):
        start_idx = i * window_len
        end_idx = min(start_idx + window_len, n)
        window_data = df.iloc[start_idx:end_idx]

        avg_row = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(window_data[col]):
                avg_row[col] = window_data[col].mean()
            else:
                # đặt 'date' là bắt đầu window
                avg_row[col] = window_data['date'].iloc[0] if 'date' in window_data.columns else window_data.iloc[0][col]

        # thêm metadata window (bắt đầu và kết thúc) để dễ mapping sau này
        avg_row['_window_start'] = window_data['date'].iloc[0]
        avg_row['_window_end'] = window_data['date'].iloc[-1]

        result_rows.append(avg_row)

    avg_df = pd.DataFrame(result_rows)
    avg_df['date'] = pd.to_datetime(avg_df['date'])
    avg_df = avg_df.sort_values('date').reset_index(drop=True)
    return avg_df


def plot_anomaly(json_path, field, full_data, target_day, node, save_dir="plot_pics"):
    """
    Vẽ biểu đồ giá trị theo thời gian của 'field' trong 'full_data'
    và highlight các điểm anomaly đọc từ file JSON.
    """
    import os
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    os.makedirs(save_dir, exist_ok=True)

    # --- Đọc dữ liệu anomaly từ file JSON ---
    with open(json_path, "r", encoding="utf-8") as f:
        anomalies = pd.DataFrame(json.load(f))

    anomalies["date"] = pd.to_datetime(anomalies["date"])
    anomalies = anomalies[(anomalies["node"] == node) & (anomalies["anomaly"] == -1)]

    # --- Lọc dữ liệu chính ---
    df = full_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["node"] == node) & (df["date"].dt.strftime("%Y-%m-%d") == target_day)]

    if df.empty:
        print(f"[plot_anomaly] ⚠️ Không có dữ liệu cho {field} node {node} tại {target_day}")
        return None

    # --- Lọc anomaly trong ngày ---
    anomaly_points = anomalies[
        (anomalies["date"].dt.strftime("%Y-%m-%d") == target_day)
    ]

    # --- Vẽ ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df[field], color="#1f77b4", lw=2, label="Giá trị đo")
    ax.set_title(f"{field} - Node {node} ({target_day})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Giá trị")

    # --- Highlight anomaly ---
    if not anomaly_points.empty:
        merged = pd.merge(df, anomaly_points[["date", "score"]], on="date", how="inner")
        ax.scatter(
            merged["date"], merged[field],
            color="red", s=60, label="Anomaly", zorder=5, alpha=0.8, edgecolors="black"
        )

    # --- Định dạng trục thời gian ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # --- Lưu ảnh ---
    img_path = os.path.join(save_dir, f"{field}_node{node}_{target_day}.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    print(f"[plot_anomaly] ✅ Đã lưu {img_path}")
    return img_path



def create_json_output(answer, json_path="result_ml.json", output_path="result_structured.json"):
    import json
    import pandas as pd

    # --- Load dữ liệu anomaly thật ---
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    structured = []
    anomaly_counter = 1

    try:
        # Tách các dòng không rỗng
        lines = [l.strip() for l in answer.splitlines() if l.strip()]

        # Tìm vị trí dòng bắt đầu chứa JSON (bắt đầu bằng "[" hoặc "{")
        start_idx = next((i for i, l in enumerate(lines) if l.startswith("[") or l.startswith("{")), None)
        if start_idx is not None:
            json_text = "\n".join(lines[start_idx:])
            # Cắt phần còn lại nếu sau JSON có chữ, đảm bảo đóng ngoặc đủ
            try:
                # Dò thủ công số ngoặc mở/đóng để xác định phần JSON đầy đủ
                open_brackets = 0
                json_lines = []
                for line in lines[start_idx:]:
                    open_brackets += line.count("[") + line.count("{")
                    open_brackets -= line.count("]") + line.count("}")
                    json_lines.append(line)
                    if open_brackets == 0:
                        break
                json_text = "\n".join(json_lines)
            except Exception:
                pass

            # Parse JSON
            json_data = json.loads(json_text)

            # Chuẩn hóa về dạng anomaly list
            structured = []
            for i, item in enumerate(json_data, start=1):
                structured.append({
                    f"anomaly_{i}": {
                        "name_field": [item.get("field")],
                        "content": item.get("reason", ""),
                        "meta": {
                            "date": item.get("date"),
                            "node": item.get("node"),
                        }
                    }
                })
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, ensure_ascii=False, indent=2)
            print(f" Đã tạo file JSON có cấu trúc: {output_path}")
            print(f"   → Tổng số anomaly: {len(structured)}")
            return structured

    except Exception as e:
        print(f" Không phát hiện JSON trong kết quả, fallback sang parser cũ: {e}")


        # --- TH2: fallback sang logic markdown cũ (giữ nguyên của bạn) ---
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        current_date = None
        current_item = None
        result = []

        for line in lines:
            if line.startswith("### Phân tích bất thường mạng ngày"):
                current_date = line.split()[-1]
                continue
            if line.startswith("#### Khoảng thời gian"):
                if current_item:
                    result.append(current_item)
                current_item = {
                    "date": current_date,
                    "window_text": line.replace("#### ", "").strip(),
                    "node": None,
                    "score": None,
                    "fields": [],
                    "content": ""
                }
                continue
            if line.startswith("- Node"):
                try:
                    current_item["node"] = int(line.split(":")[1].strip())
                except Exception:
                    pass
                continue
            if line.startswith("- Score"):
                try:
                    current_item["score"] = float(line.split(":")[1].strip())
                except Exception:
                    pass
                continue
            if line.startswith("-"):
                match = re.search(r"[-•]\s*(?:`|\*\*)([A-Za-z0-9_ ]+)(?:`|\*\*)\s*:\s*([\d\.]+)", line)
                if match:
                    field_name = match.group(1).strip()
                    try:
                        value = float(match.group(2))
                    except Exception:
                        value = None
                    current_item["fields"].append((field_name, value))
                    continue
            if line.lower().startswith("nguyên nhân"):
                current_item["content"] += " " + line.strip()
                continue
            if current_item:
                current_item["content"] += " " + line.strip()

        if current_item:
            result.append(current_item)

        for item in result:
            node = item["node"]
            score = item["score"]
            date = item["date"]
            fields = [f for f, _ in item["fields"]]
            related = df[(df["node"] == node) & (df["score"].round(6) == round(score or 0, 6))]
            meta = related.iloc[0].to_dict() if not related.empty else {}
            structured.append({
                f"anomaly_{anomaly_counter}": {
                    "name_field": fields,
                    "content": item["content"].strip(),
                    "meta": meta
                }
            })
            anomaly_counter += 1

    # --- Ghi file kết quả ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f" Đã tạo file JSON có cấu trúc: {output_path}")
    print(f"   → Tổng số anomaly: {len(structured)}")

    return structured
