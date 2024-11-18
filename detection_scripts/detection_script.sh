#!/bin/bash
#use sudo chmod +x /home/dasl/code/audioRecorder/recorderLaunch.sh
#in terminal to make this file executable. Otherwise you'll get
#a permission denied error.

source /home/dasl/repos/boring_recorder/.venv/bin/activate
sudo python3 /home/dasl/repos/boring_recorder/real_time_detection.py