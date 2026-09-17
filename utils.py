import copy
import os.path
import pickle
import sys
from collections import defaultdict
from math import radians, cos, sin, asin, sqrt

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Testdataset


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def calculate_laplacian_matrix(adj_mat):
    n_vertex = adj_mat.shape[0]
    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))#增加自循环
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_mat + id_mat
    hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)#度矩阵的逆矩阵，$\hat{A} = \tilde{D}^{-1}\tilde{A}$
    return hat_rw_normd_lap_mat


def compute_time(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


# def time_interval_mat(user_train, usernum, maxlen, time_span):
#     data_train = dict()
#     for user in tqdm(range(1, usernum + 1)):
#         time_seq = np.zeros([maxlen], dtype=np.int32)
#         idx = maxlen - 1
#         for i in reversed(user_train[user][:-1]):
#             time_seq[idx] = i[1]
#             idx -= 1
#             if idx == -1:
#                 break
#         data_train[user] = compute_time(time_seq, time_span)
#     return data_train

def time_interval_mat(user_train, maxlen, time_span):
    # 1. 彻底抛弃字典！现在我们要返回一个极其庞大的列表，严格对应 sample_idx
    time_matrices_list = []

    for user, history in tqdm(user_train.items(), desc="Computing Time Matrices"):
        n_history = len(history)
        if n_history < 2:
            continue  # 至少要有 1个输入 + 1个目标 才能预测

        user_chunk_matrices = []

        # 极其冷血的逆向滑动窗口，一模一样的指针！
        for i in range(n_history, 1, -maxlen):
            start_idx = max(0, i - (maxlen + 1))
            chunk = history[start_idx: i]

            if len(chunk) > 1:
                # 2. 准备 100 长度的空一维数组，用于装填当前切片的时间戳
                time_seq = np.zeros([maxlen], dtype=np.int32)
                idx = maxlen - 1

                # 3. 极其精准的倒序装填！注意：剥离最后一个目标点 (chunk[:-1])
                for item in reversed(chunk[:-1]):
                    time_seq[idx] = item[1]  # item[1] 就是物理时间戳
                    idx -= 1
                    if idx == -1:
                        break

                # 4. 核爆点：拿着这 100 个极其纯净的时间戳，召唤原作者的 compute_time！
                # 它会吐出一个 maxlen * maxlen (100x100) 的完美矩阵
                chunk_time_mat = compute_time(time_seq, time_span)

                # 存入当前用户的临时队列
                user_chunk_matrices.append(chunk_time_mat)

        # 5. 命运的翻转：必须和 Dataset 里一样，翻转回正向时间顺序！
        user_chunk_matrices.reverse()

        # 6. 汇入全局大军！
        time_matrices_list.extend(user_chunk_matrices)

    return time_matrices_list


def compute_dist(dis_seq, dis_span):
    dis_span = dis_span
    size = len(dis_seq)
    dis_matrix = np.zeros([size, size], dtype=np.float64)
    for i in range(size):
        for j in range(size):
            lon1 = float(dis_seq[i].split(',')[0])
            lat1 = float(dis_seq[i].split(',')[1])
            lon2 = float(dis_seq[j].split(',')[0])
            lat2 = float(dis_seq[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))
            if dis_seq[i] == '0,0' or dis_seq[j] == '0,0':
                dis_matrix[i][j] = dis_span
            elif span > dis_span:
                dis_matrix[i][j] = dis_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix



def dist_interval_mat(user_train, maxlen, dis_span):
    # 1. 彻底抛弃字典，转为全局切片列表！
    dist_matrices_list = []
    for user, history in tqdm(user_train.items(), desc="Computing Distance Matrices"):
        n_history = len(history)
        if n_history < 2:
            continue  # 至少要有 1个输入 + 1个目标 才能预测

        user_chunk_matrices = []

        # 极其冷血的逆向滑动窗口指针（和之前绝对一模一样！）
        for i in range(n_history, 1, -maxlen):
            start_idx = max(0, i - (maxlen + 1))
            chunk = history[start_idx: i]

            if len(chunk) > 1:
                # 2. 准备空坑位：空间矩阵的 Padding 幽灵必须是字符串 '0,0'！
                dis_seq = ['0,0'] * maxlen
                idx = maxlen - 1

                # 3. 极其精准的倒序装填！(依旧剥离最后一个目标点 chunk[:-1])
                for item in reversed(chunk[:-1]):
                    # 💡 注意这里的下标是 2！
                    # 因为历史轨迹的元组里，item[0] 是 POI ID，item[1] 是时间戳，item[2] 是 "经度,纬度" 字符串！
                    dis_seq[idx] = item[2]
                    idx -= 1
                    if idx == -1:
                        break

                # 4. 核爆点：调用咱们之前重写的那个“绝对忠于原论文”的 compute_dist 算距离！
                # 它会把这 100 个经纬度，变成一个 100x100 的整数分类矩阵
                chunk_dis_mat = compute_dist(dis_seq, dis_span)

                user_chunk_matrices.append(chunk_dis_mat)

        # 5. 命运的翻转：保持和 Dataset 以及 time_mat 翻转逻辑绝对一致！
        user_chunk_matrices.reverse()

        # 6. 汇入全局大军！
        dist_matrices_list.extend(user_chunk_matrices)

    return dist_matrices_list


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time - time_min)))
    return time_map


def clean_sort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u + 1
    for i, item in enumerate(item_set):
        item_map[item] = i + 1

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list) - 1):
            if time_list[i + 1] - time_list[i] != 0:
                time_diff.add(time_list[i + 1] - time_list[i])
        if len(time_diff) == 0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1), x[2]], items))#时间是本次-最小，再除以最小间隔
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)
# User_res = {
#     1: [
#         [1, 1, 'LocB'], # 原 99 映射为 1。时间步 1
#         [2, 2, 'LocA'], # 原 88 映射为 2。距离上次间隔2分钟，相对时间步为 2
#         [2, 4, 'LocA']  # 再次访问 88。距离上次4分钟(是最小间隔2分钟的2倍)，时间步为 4
#     ]
# }


def data_partition(fname):
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}

    print('Starting data partition on {}.'.format(fname.split('/')[1]))
    f = open(fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u, i = int(u), int(i)
        user_count[u] += 1
        item_count[i] += 1
    f.close()
    f = open(fname, 'r')

    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        time_set.add(timestamp)
        User[u].append([i, timestamp, location])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = clean_sort(User, time_map)
    user_poi_dict = {user: list(set([item[0] for item in records])) for user, records in User.items()}

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            # user_train[user] = User[user][:-2]
            # user_valid[user] = []
            # user_valid[user].append(User[user][-2])
            # user_test[user] = []
            # user_test[user].append(User[user][-1])
            # 极其冷血的 8:1:1 物理切割指针
            train_idx = int(nfeedback * 0.8)
            valid_idx = int(nfeedback * 0.9)

            # 极简防爆护盾：保证验证和测试集至少有1个元素
            if valid_idx == train_idx:
                train_idx -= 1
            if valid_idx == nfeedback:
                valid_idx -= 1
                if valid_idx == train_idx:
                    train_idx -= 1

            # 命运交割：按时间轴切断！
            user_train[user] = User[user][:train_idx]
            user_valid[user] = User[user][train_idx:valid_idx]
            user_test[user] = User[user][valid_idx:]

    pop_freq = [0] * (itemnum+1)
    for u in user_train:
        history = user_train[u]
        for elem in history:
            pop_freq[elem[0]] += 1
    # pop_freq[0] = 1e-5
    return [user_train, user_valid, user_test, usernum, itemnum, timenum, pop_freq, user_poi_dict]


def sparse_dropout(x, rate, noise_shape):
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1. / (1 - rate))

    return out


def get_adj_matrix(matrix):
    row_sum = np.array(matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(matrix.dot(degree_mat_inv_sqrt)).todense()
    return rel_matrix_normalized

def generate_test(dataset, args):
    [train, valid, test, usernum, _, _, _,_] = copy.deepcopy(dataset)
    users = range(1, usernum + 1)
    all_test_user, all_test_seq, all_test_time_matrix, all_test_dis_matrix, all_labels, all_test_lens = [], [], [], [], [], []


    for u in tqdm(users, desc="Processing Test Users"):
        # 1. 幽灵防御：如果该用户根本没有测试集，直接无视
        if not test[u]:
            continue

        # 2. 极其粗暴且绝对准确地拼接用户“一生的轨迹”
        # 把被 8:1:1 物理切断的轨迹重新连起来，方便我们滑动截取！
        full_history = train[u] + valid[u] + test[u]

        # 3. 找到测试集在“一生轨迹”里的绝对起跑线
        test_start_idx = len(train[u]) + len(valid[u])

        # 4. 🔥 核心核爆：遍历测试集里的【每一个】目标点！
        for k in range(test_start_idx, len(full_history)):
            target_item = full_history[k]
            label = target_item[0]  # 这个就是要预测的真实 Next POI 标签！

            # 5. 截取目标点之前的、最多 maxlen 个历史轨迹作为输入
            start_idx = max(0, k - args.maxlen)
            past_seq = full_history[start_idx: k]

            # 如果历史是空的（极端情况），无法预测，跳过
            if not past_seq:
                continue

            true_length = len(past_seq)

            # 6. 准备极其干净的空坑位（Padding 幽灵阵列）
            seq = np.zeros([args.maxlen], dtype=np.int32)
            time_seq = np.zeros([args.maxlen], dtype=np.int32)
            dis_seq = ['0,0'] * args.maxlen

            # 7. 极其精准的右对齐倒序装填！
            idx = args.maxlen - 1
            for item in reversed(past_seq):
                seq[idx] = item[0]  # POI
                time_seq[idx] = item[1]  # 时间
                dis_seq[idx] = item[2]  # 空间经纬度
                idx -= 1
                if idx == -1:
                    break

            # 8. 召唤外挂矩阵计算引擎！
            time_matrix = compute_time(time_seq, args.time_span)
            dis_matrix = compute_dist(dis_seq, args.dis_span)

            # 9. 满载弹药，打包装车！
            all_test_user.append(u)
            all_test_seq.append(seq)
            all_test_time_matrix.append(time_matrix)
            all_test_dis_matrix.append(dis_matrix)
            all_labels.append(label)
            all_test_lens.append(true_length)

    # 10. 保存极其庞大的多目标测试集
    with open(args.dataset + '_test_instance.pkl', 'wb') as f:
        pickle.dump(all_test_user, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_seq, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_time_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_dis_matrix, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_labels, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(all_test_lens, f, pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ 测试集生成完毕！原版只有 {usernum} 个测试样本，现在暴增至: {len(all_labels)} 个！")
def evaluate_test(model, dataset, args,user_poi_dict,itemnum):
    HT, NDCG = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    global_mrr = 0.0  # 极其纯粹的全局 MRR 累加器
    test_user_num = 0.0
    test_pkl_path = args.dataset + "_test_instance.pkl"
    if not os.path.exists(test_pkl_path):
        print('Preparing test instances...')
        generate_test(dataset, args)
    with open(test_pkl_path, 'rb') as f:
        all_u = pickle.load(f)
        all_seqs = pickle.load(f)
        all_time_matrix = pickle.load(f)
        all_distance_matrix = pickle.load(f)
        all_labels = pickle.load(f)
        all_lens = pickle.load(f)

    test_dataset = Testdataset(all_u, all_seqs, all_time_matrix, all_distance_matrix, all_labels,user_poi_dict,itemnum,all_lens)
    dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for instance in dataloader:
            sys.stdout.flush()
            u, seq, time_matrix, dis_matrix, label,exclude_mask, seq_lens = instance
            predictions = model.predict(u, seq, time_matrix, dis_matrix, seq_lens)
            label = label.to(args.device)#修改
            exclude_mask = exclude_mask.to(args.device)
            predictions[:, 0] = -1e10
            predictions[exclude_mask] = -1e10
            label_scores = predictions.gather(1, label.view(-1, 1))
            rank = (predictions > label_scores).sum(dim=1).detach().cpu().tolist()
            test_user_num += len(rank)
            for i in rank:
                if i < 2:
                    NDCG[0] += 1 / np.log2(i + 2)
                    HT[0] += 1
                if i < 5:
                    NDCG[1] += 1 / np.log2(i + 2)
                    HT[1] += 1
                if i < 10:
                    NDCG[2] += 1 / np.log2(i + 2)
                    HT[2] += 1
                # ---------------------------------------------------
                # 极其高贵的全局统筹派：MRR 直接按真实物理位置算分！
                # ---------------------------------------------------
                global_mrr += 1.0 / (i + 1)

    # final_MRR = global_mrr / test_user_num
    final_mrr = float(global_mrr / test_user_num)

    return [float(x / test_user_num) for x in NDCG], [float(x / test_user_num) for x in HT], [final_mrr]
