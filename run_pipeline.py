"""run_pipeline_admin.py - Run with admin privileges and clean directories"""
import os
import shutil
import subprocess
import sys

# Paths
project_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk_spark"
data_dir = r"D:\Projects\Major Project\freddie_mac_credit_risk\data"

# Kill any Spark processes
os.system("taskkill /f /im java.exe 2>nul")
os.system("taskkill /f /im python.exe 2>nul")

# Wait a moment
import time
time.sleep(2)

# Delete problematic directories
for dir_name in ["features", "splits", "models"]:
    dir_path = os.path.join(data_dir, dir_name)
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Deleted: {dir_path}")
        except Exception as e:
            print(f"Could not delete {dir_path}: {e}")

# Run the pipeline
os.chdir(project_dir)
subprocess.run([sys.executable, "pipeline_spark.py", "--stages", "features"])