from flask import Flask, render_template, jsonify
from trainnewdt import process_and_predict
from telegram_notifier import TelegramNotifier
from database_handler import DatabaseHandler
import json
import time
from datetime import datetime

app = Flask(__name__)

# Khởi tạo các handlers
telegram_notifier = TelegramNotifier()
db_handler = DatabaseHandler()

# Biến lưu trữ trạng thái máy bơm
pump_status = {
    "pump1": {"status": "Unknown", "last_updated": None},
    "pump2": {"status": "Not in use", "last_updated": None},
    "pump3": {"status": "Not in use", "last_updated": None},
    "pump4": {"status": "Not in use", "last_updated": None}
}

@app.route('/')
def index():
    # Lấy lịch sử dự đoán gần đây cho máy bơm 1
    recent_predictions = db_handler.get_recent_predictions("pump1", limit=5)
    return render_template('index.html', 
                         pump_status=pump_status,
                         prediction_history=recent_predictions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Thực hiện dự đoán
        prediction_result = process_and_predict()
        if prediction_result is not None:
            # Cập nhật trạng thái máy bơm 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pump_status["pump1"]["status"] = prediction_result
            pump_status["pump1"]["last_updated"] = current_time
            
            # Gửi thông báo qua Telegram
            telegram_notifier.send_notification(prediction_result)
            
            # Lưu kết quả vào database
            db_handler.save_prediction(
                pump_id="pump1",
                status=prediction_result,
                features=None,  # Có thể thêm features nếu cần
                confidence=None  # Có thể thêm confidence nếu có
            )
            
            return jsonify({
                "success": True,
                "prediction": prediction_result,
                "pump_status": pump_status,
                "timestamp": current_time
            })
        else:
            return jsonify({
                "success": False,
                "error": "Prediction failed"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 