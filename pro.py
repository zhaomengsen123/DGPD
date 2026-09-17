import pandas as pd
import numpy as np
import scipy.sparse as sp
from haversine import haversine
from tqdm import tqdm
import os

# ================= 配置区 =================
# 替换为你的原始文件名
# RAW_DATA_PATH = 'dataset_TSMC2014_NYC.txt'
RAW_DATA_PATH = 'dataset_TSMC2014_TKY.txt'
DATASET_NAME = 'tky'
# RAW_DATA_PATH = 'dataset_TSMC2014_TKY.txt'
# DATASET_NAME = 'tky'
OUTPUT_DIR = f'data/{DATASET_NAME}'
DISTANCE_THRESHOLD = 0.5
# ==========================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("1. 读取 Foursquare 格式数据...")
# 定义 Foursquare 数据的列名
columns = ['user_id', 'venue_id', 'cat_id', 'cat_name', 'latitude', 'longitude', 'timezone_offset', 'utc_time']

# 按 tab 键分割读取
df = pd.read_csv(RAW_DATA_PATH, sep='\t', header=None, names=columns, encoding='unicode_escape')


print("2. 转换时间戳...")
# 将英文时间 "Tue Apr 03 18:00:09 +0000 2012" 转换为 Unix 时间戳
df['timestamp'] = pd.to_datetime(df['utc_time'], format='%a %b %d %H:%M:%S +0000 %Y').astype(np.int64) // 10 ** 9

print("3. 进行 K-core (5次) 过滤...")
# 循环过滤交互少于 5 次的用户和 POI
while True:
    user_counts = df['user_id'].value_counts()
    poi_counts = df['venue_id'].value_counts()

    valid_users = user_counts[user_counts >= 5].index
    valid_pois = poi_counts[poi_counts >= 5].index

    df_filtered = df[df['user_id'].isin(valid_users) & df['venue_id'].isin(valid_pois)]
    if len(df_filtered) == len(df):
        break
    df = df_filtered

print("4. 重新编码 ID (从 1 开始，0 用于 Padding)...")
user_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(df['user_id'].unique())}
poi_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(df['venue_id'].unique())}

df['user_id'] = df['user_id'].map(user_mapping)
df['poi_id'] = df['venue_id'].map(poi_mapping)
df = df.sort_values(by=['user_id', 'timestamp'])

num_users = len(user_mapping)
num_pois = len(poi_mapping)
print(f"有效用户数: {num_users}, 有效 POI 数: {num_pois}")
print(df)

# # 1. 找到整个数据集（或者每个用户）的最早时间戳作为基准 0 点
# min_timestamp = df['timestamp'].min()
#
# # 2. 减去基准点，算出差值（秒），然后极其果断地除以 60 转换为“分钟”！
# # 加上 .astype(int) 彻底抹杀小数，变成大模型最爱的整型 ID
# df['timestamp'] = ((df['timestamp'] - min_timestamp) / 60).astype(int)
# print(df)

print("5. 生成 mydata.txt...")
txt_path = f"{OUTPUT_DIR}/{DATASET_NAME}.txt"
with open(txt_path, 'w') as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 严格按照 utils.py 要求的格式: user \t item \t lon,lat \t timestamp
        line = f"{int(row['user_id'])}\t{int(row['poi_id'])}\t{row['longitude']},{row['latitude']}\t{row['timestamp']}\n"
        f.write(line)

print("6. 构建转移矩阵 (Transition Matrix)...")
# tran_mat = np.zeros((num_pois + 1, num_pois + 1), dtype=np.float32)
#
# for user_id, group in tqdm(df.groupby('user_id')):
#     poi_seq = group['poi_id'].tolist()
#     for i in range(len(poi_seq) - 1):
#         source = poi_seq[i]
#         target = poi_seq[i + 1]
#         tran_mat[source][target] += 1
#
# sparse_tran_mat = sp.coo_matrix(tran_mat)
# # print(tran_mat)
# # print(sparse_tran_mat)
# sp.save_npz(f"{OUTPUT_DIR}/{DATASET_NAME}_mat.npz", sparse_tran_mat)
# np.savez(f"{OUTPUT_DIR}/{DATASET_NAME}_tran_mat.npz", data=tran_mat)
# #
# print("7. 构建地理距离矩阵 (Geography Matrix)...")
# dist_mat = np.zeros((num_pois + 1, num_pois + 1), dtype=np.float32)
#
# poi_coords = {}
# for _, row in df.drop_duplicates(subset=['poi_id']).iterrows():
#     poi_coords[int(row['poi_id'])] = (row['latitude'], row['longitude'])
#
# pois = list(poi_coords.keys())
# for i in tqdm(range(len(pois))):
#     for j in range(i + 1, len(pois)):
#         p1 = pois[i]
#         p2 = pois[j]
#         distance = haversine(poi_coords[p1], poi_coords[p2])
#         if distance <= DISTANCE_THRESHOLD:
#             dist_mat[p1][p2] = 1.0
#             dist_mat[p2][p1] = 1.0
# print(dist_mat)
#
# np.savez(f"{OUTPUT_DIR}/{DATASET_NAME}_dist_mat.npz", data=dist_mat)
#
# print("所有预处理文件生成完毕！现在可以运行 main.py 了。")