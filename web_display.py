from flask import Flask, render_template, jsonify, send_file
from database_handler import DatabaseHandler
import json
from datetime import datetime, timedelta
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

def get_pump_status():
    db = DatabaseHandler()
    latest_predictions = db.get_recent_predictions(limit=1)
    
    status = {
        'pump1': {
            'status': 'Unknown',
            'last_updated': None
        }
    }
    
    if latest_predictions:
        latest = latest_predictions[0]
        status['pump1'] = {
            'status': latest['status'].upper(),
            'last_updated': latest['time']
        }
    
    return status

@app.route('/')
def index():
    db = DatabaseHandler()
    latest_predictions = db.get_recent_predictions(limit=10)
    pump_status = get_pump_status()
    return render_template('index.html', predictions=latest_predictions, pump_status=pump_status)

@app.route('/api/latest')
def get_latest():
    db = DatabaseHandler()
    latest_predictions = db.get_recent_predictions(limit=10)
    return json.dumps(latest_predictions, cls=DateTimeEncoder)

@app.route('/api/status')
def get_status():
    status = get_pump_status()
    return jsonify(status)

@app.route('/api/daily-stats')
def get_daily_stats():
    db = DatabaseHandler()
    stats = db.get_daily_stats(days=7)
    return json.dumps(stats, cls=DateTimeEncoder)

@app.route('/api/heatmap-data')
def get_heatmap_data():
    db = DatabaseHandler()
    data = db.get_hourly_heatmap(days=7)
    return json.dumps(data, cls=DateTimeEncoder)

@app.route('/api/export-csv')
def export_csv():
    try:
        db = DatabaseHandler()
        # Lấy dữ liệu 7 ngày gần nhất
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        predictions = db.get_predictions_by_timerange(start_time, end_time)
        
        # Chuyển thành DataFrame
        df = pd.DataFrame(predictions)
        
        # Tạo buffer để lưu file CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return send_file(
            io.BytesIO(buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'pump_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-report')
def export_report():
    try:
        db = DatabaseHandler()
        predictions = db.get_recent_predictions(limit=1000)
        stats = db.get_daily_stats(days=7)
        heatmap_data = db.get_hourly_heatmap(days=7)

        # Tạo DataFrame cho heatmap
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_table = heatmap_df.pivot(index='date', columns='hour', values='fault_count')

        # Tạo PDF report với matplotlib
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Heatmap
        plt.subplot(2, 1, 1)
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
        plt.title('Fault Occurrence Heatmap')
        
        # Plot 2: Daily Stats
        stats_df = pd.DataFrame(stats)
        plt.subplot(2, 1, 2)
        stats_df.plot(x='date', y=['fault_rate', 'avg_normal_prob', 'avg_fault_prob'])
        plt.title('Daily Statistics')
        
        # Lưu vào buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='pdf', bbox_inches='tight')
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'pump_report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 