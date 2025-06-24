import os
import shutil

# 设置路径
txt_file = '/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/dgain_split_file/dgain_50_100.txt'                 # 包含文件名的txt，每行一个，例如：img001.jpg
source_dir = '/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/dgain_info'      # 原始文件所在目录
target_dir = '/home/ubuntu/data_sota_disk/dataset/inverse_tone/AIM_ITM_TRAIN/train_hdr/dgain_info'      # 目标目录，若不存在将自动创建

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 读取文件名列表
with open(txt_file, 'r') as f:
    base_names = [os.path.splitext(line.strip())[0] for line in f if line.strip()]

# 遍历并复制文件
for filename in base_names:
    src_path = os.path.join(source_dir, filename+'.txt')
    dst_path = os.path.join(target_dir, filename+'.txt')
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)
        print(f'Copied: {filename}')
    else:
        print(f'File not found: {filename}')
