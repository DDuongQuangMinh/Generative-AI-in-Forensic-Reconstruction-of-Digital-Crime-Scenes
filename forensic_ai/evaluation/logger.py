import csv
import os
from datetime import datetime

LOG_FILE = "results_log.csv"

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "vae_error",
                "gan_raw",
                "diff_error",
                "vae_score",
                "gan_score",
                "diff_score",
                "final_score",
                "decision"
            ])

def log_result(data):
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data["vae_error"],
            data["gan_raw"],
            data["diff_error"],
            data["vae_score"],
            data["gan_score"],
            data["diff_score"],
            data["final_score"],
            data["decision"]
        ])