import mysql.connector
from datetime import datetime, timedelta
import json

class DatabaseHandler:
    def __init__(self):
        # Thông tin kết nối MySQL
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',  # Thay đổi mật khẩu nếu có
            'database': 'pump_monitoring'
        }
        self.init_database()

    def init_database(self):
        """Khởi tạo database và bảng nếu chưa tồn tại"""
        try:
            # Kết nối đến MySQL Server
            conn = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password']
            )
            cursor = conn.cursor()

            # Tạo database nếu chưa tồn tại
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}")
            cursor.execute(f"USE {self.config['database']}")

            # Tạo bảng predictions với cấu trúc mới
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    time DATETIME,
                    status VARCHAR(50),
                    normal_prob FLOAT,
                    fault_prob FLOAT,
                    sensor_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            conn.commit()
            cursor.close()
            conn.close()
            print("✅ Khởi tạo database thành công")

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi khởi tạo database: {err}")

    def save_prediction(self, status, normal_prob, fault_prob, sensor_data=None):
        """Lưu kết quả dự đoán vào database"""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor()

            query = """
                INSERT INTO predictions 
                (time, status, normal_prob, fault_prob, sensor_data)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # Chuyển sensor_data thành JSON string nếu có
            sensor_data_json = json.dumps(sensor_data) if sensor_data else None
            
            values = (
                datetime.now(),
                status,
                normal_prob,
                fault_prob,
                sensor_data_json
            )
            
            cursor.execute(query, values)
            conn.commit()
            
            cursor.close()
            conn.close()
            return True

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi lưu dự đoán: {err}")
            return False

    def get_recent_predictions(self, limit=10):
        """Lấy các dự đoán gần đây nhất"""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT * FROM predictions 
                ORDER BY time DESC 
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()

            # Chuyển đổi JSON string thành dict
            for row in results:
                if row['sensor_data']:
                    row['sensor_data'] = json.loads(row['sensor_data'])

            cursor.close()
            conn.close()
            return results

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi truy vấn dữ liệu: {err}")
            return []

    def get_predictions_by_timerange(self, start_time, end_time):
        """Lấy dự đoán trong khoảng thời gian"""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT * FROM predictions 
                WHERE time BETWEEN %s AND %s
                ORDER BY time ASC
            """
            cursor.execute(query, (start_time, end_time))
            results = cursor.fetchall()

            for row in results:
                if row['sensor_data']:
                    row['sensor_data'] = json.loads(row['sensor_data'])

            cursor.close()
            conn.close()
            return results

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi truy vấn dữ liệu: {err}")
            return []

    def get_daily_stats(self, days=7):
        """Lấy thống kê theo ngày"""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT 
                    DATE(time) as date,
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN status = 'fault' THEN 1 ELSE 0 END) as fault_count,
                    AVG(CASE WHEN status = 'fault' THEN 1 ELSE 0 END) as fault_rate,
                    AVG(normal_prob) as avg_normal_prob,
                    AVG(fault_prob) as avg_fault_prob
                FROM predictions 
                WHERE time >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY)
                GROUP BY DATE(time)
                ORDER BY date DESC
            """
            cursor.execute(query, (days,))
            results = cursor.fetchall()

            cursor.close()
            conn.close()
            return results

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi truy vấn thống kê: {err}")
            return []

    def get_hourly_heatmap(self, days=7):
        """Lấy dữ liệu cho heatmap theo giờ"""
        try:
            conn = mysql.connector.connect(**self.config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT 
                    HOUR(time) as hour,
                    DATE(time) as date,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN status = 'fault' THEN 1 ELSE 0 END) as fault_count
                FROM predictions 
                WHERE time >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY)
                GROUP BY DATE(time), HOUR(time)
                ORDER BY date DESC, hour ASC
            """
            cursor.execute(query, (days,))
            results = cursor.fetchall()

            cursor.close()
            conn.close()
            return results

        except mysql.connector.Error as err:
            print(f"❌ Lỗi khi truy vấn dữ liệu heatmap: {err}")
            return [] 