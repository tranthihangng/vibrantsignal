import requests
from datetime import datetime

class TelegramNotifier:
    def __init__(self):
        self.BOT_TOKEN = '7831027284:AAHC7qNuD_Iq7-xJLeUh92zdhiASR0T33_U'
        self.CHAT_ID = '-4531018311'
        self.BASE_URL = f'https://api.telegram.org/bot{self.BOT_TOKEN}/sendMessage'
        self.last_status = None  # Để tránh gửi thông báo trùng lặp

    def send_notification(self, status):
        # Chỉ gửi thông báo khi trạng thái thay đổi
        if status == self.last_status:
            return
            
        self.last_status = status
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo emoji và message tương ứng với từng trạng thái
        status_info = {
            'stop': ('🛑', 'Động cơ dừng'), 
            'normal': ('✅', 'Máy bơm đang hoạt động bình thường'),
            'rung_6': ('⚠️', 'Cảnh báo: Động cơ rung nhẹ'),
            'rung_12_5': ('🔥', 'NGUY HIỂM! Máy bơm rung mạnh – cần kiểm tra NGAY!')
        }

        emoji, message = status_info.get(status, ('❓', 'Trạng thái không xác định'))
        
        # Tạo nội dung thông báo
        notification_text = f"{emoji} {message}\n⏰ Thời gian: {current_time}\n🔄 Trạng thái: {status}"
        
        # Gửi thông báo
        payload = {
            'chat_id': self.CHAT_ID,
            'text': notification_text,
            'parse_mode': 'HTML'
        }
        
        try:
            response = requests.post(self.BASE_URL, data=payload)
            if response.status_code != 200:
                print(f"Lỗi khi gửi thông báo Telegram: {response.text}")
        except Exception as e:
            print(f"Lỗi khi gửi thông báo Telegram: {str(e)}") 