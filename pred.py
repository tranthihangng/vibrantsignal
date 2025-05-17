# predict_manual_input.py

import numpy as np
import joblib
import time
from datetime import datetime
import os
from database_handler import DatabaseHandler

# path = "NCKH_FI/mlp_best_model.joblib"
# if not os.path.exists(path):
#     print("File không tồn tại:", path)
# else:
#     model = joblib.load(path)
# Dùng cách an toàn để tránh lỗi unicode
# path = "D:/NCKH/NCKH_FI/mlp_best_model.joblib"

# if not os.path.exists(path):
#     print("File không tồn tại:", path)
# else:
#     model = joblib.load(path)
#     print("Tải mô hình thành công!")

# === Load mô hình và scaler đã lưu ===
model = joblib.load("mlp_best_model.joblib")
scaler = joblib.load("scaler.joblib")

print("📌 Các nhãn lớp:", model.classes_)

def get_sensor_data():
    """Hàm này sẽ lấy dữ liệu từ cảm biến. Hiện tại dùng dữ liệu mẫu"""
    return [
        370.0,395.0,401.0,427.0,400.845,8.72192495954878,-0.07123772582064253,
        -0.0038091416031678094,627.3651464895928,522.0469353743749,286.622241390283,
        724.7144115046523,55.05505505505506,44.04404404404405,51.051051051051054,
        124.12412412412414,19132.793175772706,37209.813234632566,301.0,329.0,337.0,
        369.0,336.178,10.68453630252619,-0.1982585158803512,0.14284301226963958,
        997.7886756217295,220.82917449508707,688.782628327624,516.3288002856712,
        44.04404404404405
    ]

def predict_and_save():
    """Hàm dự đoán và lưu kết quả vào database"""
    try:
        # Lấy dữ liệu từ cảm biến
        manual_input = get_sensor_data()

        # Kiểm tra số lượng đặc trưng
        if len(manual_input) != 31:
            raise ValueError(f"Số lượng đặc trưng không hợp lệ: cần 31, đang có {len(manual_input)}")

        # Chuyển thành mảng numpy và reshape
        X_input = np.array(manual_input).reshape(1, -1)

        # Chuẩn hóa và dự đoán
        X_scaled = scaler.transform(X_input)
        
        # Đo thời gian dự đoán
        start_time = time.time()
        prediction = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        end_time = time.time()

        elapsed_time_ms = (end_time - start_time) * 1000

        # Xử lý kết quả
        status = "fault" if prediction[0] == 1 else "normal"
        normal_prob = probabilities[0][0]
        fault_prob = probabilities[0][1]
        
        print(f"\n⏱️ [{datetime.now()}] Đang dự đoán...")
        print(f"➡️  Kết quả: {status} (class {prediction[0]})")
        print(f"   Xác suất: Normal={normal_prob:.3f}, Fault={fault_prob:.3f}")
        print(f"   Thời gian tính toán: {elapsed_time_ms:.3f} ms")

        # Lưu vào database
        db = DatabaseHandler()
        success = db.save_prediction(
            status=status,
            normal_prob=normal_prob,
            fault_prob=fault_prob
        )
        
        if success:
            print(f"✅ Đã lưu trạng thái vào database: {status}")
        else:
            print(f"❌ Lỗi khi lưu vào database")

        return True

    except Exception as e:
        print(f"❌ Lỗi trong quá trình dự đoán: {e}")
        return False

def run_continuous_prediction(interval=3):
    """Chạy dự đoán liên tục với khoảng thời gian interval giây"""
    print(f"🚀 Bắt đầu chạy dự đoán liên tục (interval: {interval}s)")
    print("⌛ Nhấn Ctrl+C để dừng...")
    
    try:
        while True:
            predict_and_save()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Đã dừng chương trình theo yêu cầu")

if __name__ == "__main__":
    run_continuous_prediction()


