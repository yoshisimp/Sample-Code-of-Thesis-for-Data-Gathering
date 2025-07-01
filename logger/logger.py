import os
import csv
import sqlite3
from datetime import datetime

class BehaviorLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, "behavior_log.csv")
        self.db_path = os.path.join(log_dir, "behavior_log.db")

        self.init_csv()
        self.init_db()

    def init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Name", "Emotion", "Speech", "Distractions"])

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                name TEXT,
                emotion TEXT,
                speech TEXT,
                distractions TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log(self, name, emotion, speech, distractions):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        distractions_str = ', '.join(distractions)

        # Write to CSV
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, name, emotion, speech, distractions_str])

        # Write to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO behavior (timestamp, name, emotion, speech, distractions)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, name, emotion, speech, distractions_str))
        conn.commit()
        conn.close()

