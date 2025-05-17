import numpy as np
from scipy.stats import skew, kurtosis
import os
from scipy.fft import fft
import pandas as pd
from pathlib import Path

# Các tham số
fs = 2000  # Tần số mẫu
# Các tần số đặc trưng
ftf = 50
bpfi = 120
bpfo = 90
bsf = 60

def freq2index(f, len_fft):
    """Chuyển tần số về chỉ số trong phổ FFT"""
    return int(f * len_fft / fs)

def fft_spectrum(data):
    """Tính phổ FFT của tín hiệu"""
    len_fft = len(data)
    spectrum = np.abs(fft(data))[:len_fft // 2]
    return spectrum, len_fft

def extract_axis_features(data):
    """Trích xuất đặc trưng cho một trục"""
    # Đặc trưng thống kê cơ bản
    features = list(np.percentile(data, [0, 25, 50, 100]))
    features += [np.mean(data), np.std(data), skew(data), kurtosis(data)]

    # Đặc trưng từ FFT
    fft_amps, len_fft = fft_spectrum(data)
    features += [fft_amps[freq2index(f, len_fft)] for f in [ftf, bpfi, bpfo, bsf]]

    # Top 4 tần số mạnh nhất (bỏ qua tần số cao nhất)
    n = 5
    freqs = np.linspace(0, fs/2, len(fft_amps))
    top_indices = np.argsort(fft_amps)[-n:][::-1]
    top_indices = top_indices[1:]  # bỏ tần số mạnh nhất
    features += list(freqs[top_indices])

    # Năng lượng trong các dải tần
    bands = [
        slice(freq2index(600, len_fft), len(fft_amps)),
        slice(freq2index(260, len_fft), freq2index(600, len_fft)),
    ]
    features += [np.sum(fft_amps[b]) for b in bands]

    return features

def read_vibration_file(filepath):
    """Đọc file dữ liệu rung động"""
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"⚠️ Bỏ qua file {filepath}: không có đúng 2 cột")
            return None
        return data  # Trả về toàn bộ dữ liệu 2 cột
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {filepath}: {e}")
        return None

def extract_features_from_folder(folder_path, label):
    """Trích xuất đặc trưng từ tất cả các file trong thư mục"""
    all_features_list = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Không tìm thấy thư mục: {folder_path}")
        return all_features_list

    print(f"📂 Đang xử lý thư mục: {folder_path}")
    for file in folder.glob('*.txt'):
        print(f"  📄 Đang xử lý file: {file.name}")
        data = read_vibration_file(file)
        if data is not None:
            try:
                # Trích xuất đặc trưng cho cả 2 trục
                features_x = extract_axis_features(data[:, 0])  # Trục X
                features_y = extract_axis_features(data[:, 1])  # Trục Y
                
                # Kết hợp đặc trưng của cả 2 trục
                combined_features = features_x + features_y
                combined_features.append(label)
                all_features_list.append(combined_features)
                print(f"  ✅ Trích xuất thành công: {file.name}")
            except Exception as e:
                print(f"  ❌ Lỗi khi trích xuất đặc trưng từ {file.name}: {e}")
    
    return all_features_list

def main():
    # Đường dẫn đến thư mục dts
    base_path = Path("D:/NCKH/NCKH_FI/dts")
    ok_folder = base_path / "dts_OK"
    ng_folder = base_path / "dts_NG"

    print("🚀 Bắt đầu trích xuất đặc trưng...")

    # Trích xuất đặc trưng
    ok_features = extract_features_from_folder(ok_folder, label="normal")
    ng_features = extract_features_from_folder(ng_folder, label="fault")

    if not ok_features and not ng_features:
        print("❌ Không có dữ liệu nào được trích xuất")
        return

    # Tạo tên cột cho cả 2 trục
    n_features_per_axis = len(ok_features[0]) - 1 if ok_features else len(ng_features[0]) - 1
    n_features_per_axis = n_features_per_axis // 2  # Chia 2 vì có 2 trục
    
    # Tạo tên cột cho từng trục
    columns = []
    for axis in ['X', 'Y']:
        columns.extend([f'feature_{axis}_{i+1}' for i in range(n_features_per_axis)])
    columns.append('label')

    # Tạo DataFrame
    df = pd.DataFrame(ok_features + ng_features, columns=columns)

    # Lưu file
    output_file = 'bearing_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Đã lưu đặc trưng vào file: {output_file}")
    print(f"📊 Tổng số mẫu: {len(df)}")
    print(f"   - Số mẫu normal: {len(ok_features)}")
    print(f"   - Số mẫu fault: {len(ng_features)}")
    print(f"   - Số đặc trưng mỗi trục: {n_features_per_axis}")
    print(f"   - Tổng số đặc trưng: {n_features_per_axis * 2}")

if __name__ == "__main__":
    main() 