# predict_manual_input.py

import numpy as np
import joblib
import time
from datetime import datetime
import os
from database_handler import DatabaseHandler
from flask import Flask, request, jsonify
from telegram_notifier import TelegramNotifier
import json
import threading

# Khởi tạo Flask app
app = Flask(__name__)

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
model = joblib.load("practical_mlp_best_model.joblib")
scaler = joblib.load("practical_scaler.joblib")

print("📌 Các nhãn lớp:", model.classes_)

# Khởi tạo Telegram notifier
telegram = TelegramNotifier()
last_notification_time = 0
NOTIFICATION_INTERVAL = 120  # 2 phút = 120 giây

# Biến để kiểm soát timeout
last_data_received = 0
ESP32_TIMEOUT = 50  # 50 giây timeout

# def get_sensor_data():
#     """Hàm này sẽ lấy dữ liệu từ cảm biến. Hiện tại dùng dữ liệu mẫu"""
#     return [
#         370.0,395.0,401.0,427.0,400.845,8.72192495954878,-0.07123772582064253,
#         -0.0038091416031678094,627.3651464895928,522.0469353743749,286.622241390283,
#         724.7144115046523,55.05505505505506,44.04404404404405,51.051051051051054,
#         124.12412412412414,19132.793175772706,37209.813234632566,301.0,329.0,337.0,
#         369.0,336.178,10.68453630252619,-0.1982585158803512,0.14284301226963958,
#         997.7886756217295,220.82917449508707,688.782628327624,516.3288002856712,
#         44.04404404404405
#     ]

# def predict_and_save():
#     """Hàm dự đoán và lưu kết quả vào database"""
#     try:
#         # Lấy dữ liệu từ cảm biến
#         manual_input = get_sensor_data()

#         # Kiểm tra số lượng đặc trưng
#         if len(manual_input) != 36:
#             raise ValueError(f"Số lượng đặc trưng không hợp lệ: cần 36, đang có {len(manual_input)}")

#         # Chuyển thành mảng numpy và reshape
#         X_input = np.array(manual_input).reshape(1, -1)

#         # Chuẩn hóa và dự đoán
#         X_scaled = scaler.transform(X_input)
        
#         # Đo thời gian dự đoán
#         start_time = time.time()
#         prediction = model.predict(X_scaled)
#         probabilities = model.predict_proba(X_scaled)
#         end_time = time.time()

#         elapsed_time_ms = (end_time - start_time) * 1000

#         # Xử lý kết quả
#         status = "fault" if prediction[0] == 1 else "normal"
#         normal_prob = probabilities[0][0]
#         fault_prob = probabilities[0][1]
        
#         print(f"\n⏱️ [{datetime.now()}] Đang dự đoán...")
#         print(f"➡️  Kết quả: {status} (class {prediction[0]})")
#         print(f"   Xác suất: Normal={normal_prob:.3f}, Fault={fault_prob:.3f}")
#         print(f"   Thời gian tính toán: {elapsed_time_ms:.3f} ms")

#         # Lưu vào database
#         db = DatabaseHandler()
#         success = db.save_prediction(
#             status=status,
#             normal_prob=normal_prob,
#             fault_prob=fault_prob
#         )
        
#         if success:
#             print(f"✅ Đã lưu trạng thái vào database: {status}")
#         else:
#             print(f"❌ Lỗi khi lưu vào database")

#         return True

#     except Exception as e:
#         print(f"❌ Lỗi trong quá trình dự đoán: {e}")
#         return False

# def run_continuous_prediction(interval=3):
#     """Chạy dự đoán liên tục với khoảng thời gian interval giây"""
#     print(f"🚀 Bắt đầu chạy dự đoán liên tục (interval: {interval}s)")
#     print("⌛ Nhấn Ctrl+C để dừng...")
    
#     try:
#         while True:
#             predict_and_save()
#             time.sleep(interval)
#     except KeyboardInterrupt:
#         print("\n👋 Đã dừng chương trình theo yêu cầu")

@app.route('/predict', methods=['POST'])
def receive_data():
    """Nhận dữ liệu từ client qua HTTP POST"""
    try:
        global last_data_received
        last_data_received = time.time()
        
        # Nhận dữ liệu JSON
        data = request.get_json()
        
        # Kiểm tra dữ liệu có tồn tại không
        if data is None:
            print("❌ Không nhận được dữ liệu")
            return jsonify({'error': 'No data received'}), 400

        # Kiểm tra dữ liệu là list
        if not isinstance(data, list):
            print("❌ Dữ liệu không phải là mảng")
            return jsonify({'error': 'Expected a JSON array'}), 400

        # Kiểm tra độ dài của dữ liệu (31 đặc trưng giống trong pred.py)
        expected_features = 36
        if len(data) != expected_features:
            print(f"❌ Số lượng đặc trưng không đúng: nhận {len(data)}, cần {expected_features}")
            return jsonify({'error': f'Expected {expected_features} features, got {len(data)}'}), 400

        print(f"📥 Dữ liệu nhận được từ ESP32: {data}")

        # Chuyển đổi dữ liệu thành float
        try:
            sensor_values = [float(val) for val in data]
        except (ValueError, TypeError) as e:
            print(f"❌ Lỗi định dạng dữ liệu: {str(e)}")
            return jsonify({'error': f'Invalid data format: {str(e)}'}), 400

        # Dự đoán
        prediction_result = predict_values(sensor_values)
        
        if prediction_result:
            return jsonify({
                'message': 'Prediction successful',
                'status': prediction_result['status'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities']
            }), 200
        else:
            return jsonify({'error': 'Prediction failed'}), 500

    except Exception as e:
        print(f"❌ Lỗi khi nhận dữ liệu: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/data', methods=['POST'])
def receive_data_alt():
    """Endpoint bổ sung để nhận dữ liệu từ ESP32 thông qua /data"""
    return receive_data()  # Sử dụng lại hàm xử lý của /predict

def check_esp32_timeout():
    """Hàm kiểm tra nếu ESP32 không gửi dữ liệu trong 50 giây"""
    global last_data_received
    
    while True:
        current_time = time.time()
        if last_data_received > 0:  # Nếu đã nhận dữ liệu trước đó
            time_since_last = current_time - last_data_received
            if time_since_last > ESP32_TIMEOUT:
                print(f"⚠️ ESP32 TIMEOUT: Không nhận được dữ liệu trong {ESP32_TIMEOUT} giây")
                # Có thể thêm thông báo qua Telegram nếu cần
                last_data_received = current_time  # Reset để không báo liên tục
        
        time.sleep(5)  # Kiểm tra mỗi 5 giây

def get_status_from_probabilities(probabilities):
    """Xác định trạng thái dựa trên xác suất của các lớp"""
    # Lấy index của lớp có xác suất cao nhất
    predicted_class = np.argmax(probabilities)
    prob_value = probabilities[predicted_class]
    
    # Ánh xạ index sang tên trạng thái
    status_mapping = {
        0: 'normal',
        1: 'rung_12_5',
        2: 'rung_6',
        3: 'stop'


        # 0: 'stop',
        # 1: 'normal',
        # 2: 'rung_6',
        # 3: 'rung_12_5'
    }
    
    status = status_mapping[predicted_class]
    
    # Tạo thông báo với độ tin cậy
    confidence_msg = f"{status} (độ tin cậy: {prob_value:.2%})"
    print(f"🔍 Dự đoán: {confidence_msg}")
    
    return status

def predict_values(sensor_values):
    """Hàm dự đoán từ giá trị cảm biến"""
    global last_notification_time
    
    try:
        current_time = time.time()
        time_since_last = current_time - last_notification_time

        # Chuyển thành mảng numpy và reshape
        X_input = np.array(sensor_values).reshape(1, -1)

        # Chuẩn hóa và dự đoán
        X_scaled = scaler.transform(X_input)
        
        # Đo thời gian dự đoán
        start_time = time.time()
        probabilities = model.predict_proba(X_scaled)[0]
        end_time = time.time()

        elapsed_time_ms = (end_time - start_time) * 1000

        # Xử lý kết quả
        status = get_status_from_probabilities(probabilities)
        max_prob = np.max(probabilities)
        
        print(f"⏱️ [{datetime.now()}] Đã dự đoán xong")
        print(f"   Thời gian tính toán: {elapsed_time_ms:.3f} ms")

        # Tính toán normal_prob và fault_prob từ probabilities
        normal_prob = probabilities[0] if len(probabilities) > 0 else 0
        fault_prob = probabilities[1] if len(probabilities) > 1 else 0
        
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

        # Gửi thông báo Telegram mỗi 2 phút
        if time_since_last >= NOTIFICATION_INTERVAL:
            try:
                print(f"📱 Đang gửi thông báo đến Telegram...")
                telegram.send_notification(status)
                last_notification_time = current_time
                print(f"✅ Đã gửi thông báo thành công!")
            except Exception as e:
                print(f"❌ Lỗi khi gửi thông báo Telegram: {str(e)}")

        return {
            'status': status,
            'confidence': float(max_prob),
            'probabilities': probabilities.tolist()
        }

    except Exception as e:
        print(f"❌ Lỗi trong quá trình dự đoán: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print(f"📱 Thông báo Telegram sẽ được gửi mỗi {NOTIFICATION_INTERVAL} giây")
    print(f"⏱️ Timeout ESP32: {ESP32_TIMEOUT} giây")
    
    # Khởi chạy thread kiểm tra timeout
    timeout_thread = threading.Thread(target=check_esp32_timeout, daemon=True)
    timeout_thread.start()
    
    # Chạy Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)


