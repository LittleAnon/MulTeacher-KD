import shutil 
import glob
import os
import argparse
import time

def TimeStampToTime(timestamp):
    timeStruct = time.localtime(timestamp)
    return time.strftime('%Y-%m-%d',timeStruct)


def get_FileAccessTime(filePath):
    t = os.path.getatime(filePath)
    return TimeStampToTime(t)

for log_file in glob.glob('*.log'):
    logfiletime = get_FileAccessTime(log_file)
    os.makedirs(f'./pre_log/{logfiletime}', exist_ok=True)
    shutil.move(log_file, f'./pre_log/{logfiletime}/' + os.path.split(log_file)[1])


def sort_logs():
    for log_file in glob.glob('pre_log/*.log'):
        log_file_time = get_FileAccessTime(log_file)
        # if not os.path.exists(f"pre_log/{log_file_time}"):
        os.makedirs(f"pre_log/{log_file_time}", exist_ok=True)
        shutil.move(log_file, f'./pre_log/{log_file_time}/' + os.path.split(log_file)[1])
# sort_logs()
def remove_unfinished_logs():
    pass