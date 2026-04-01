import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ivlab3/Lucas_ws/ros2_ws/install/data_collector'
