"""
function: 相对路径控制

author: chenyang wu
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 日志路径
LOG_DIR = os.path.join(BASE_DIR, '../resources/logs/info.log')