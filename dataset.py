import numpy as np
import torch
from torch.utils.data import Dataset


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class Traindataset(Dataset):
    def __init__(self, user_train, time_matrix, dis_matirx, itemnum, maxlen):
        self.user_train = user_train
        self.time_matrix = time_matrix
        self.itemnum = itemnum
        self.dis_matrix = dis_matirx
        self.max_len = maxlen

        self.samples = []

        for user, history in self.user_train.items():
            n_history = len(history)
            if n_history < 2:
                continue  # 至少要有 1个输入 + 1个目标 才能预测

            user_chunks = []
            # i 从 n_history 开始，每次往左退 100 步，直到退到序列头部
            for i in range(n_history, 1, -self.max_len):
                # i 是右边界（不包含）。计算左边界：最多往左拿 101 个点！
                start_idx = max(0, i - (self.max_len + 1))

                chunk = history[start_idx: i]
                if len(chunk) > 1:
                    user_chunks.append({
                        'user_id': user,
                        'sequence': chunk  # 长度最大为 101
                    })

            # 为了符合人类直觉，我们把它翻转回正向顺序，然后再塞入全局大样本库。因为我们是倒着切的，列表里最前面的是最新的时间段。
            user_chunks.reverse()
            self.samples.extend(user_chunks)



    def __getitem__(self, sample_idx):
        # 1. 身份剥离：彻底抛弃原版的 user_idx，现在传进来的是切片索引！
        sample = self.samples[sample_idx]
        user_id = sample['user_id']
        chunk = sample['sequence']

        # 2. 准备空阵列 (坑位填 0)
        seq = np.zeros([self.max_len], dtype=np.int32)
        time_seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)

        # 3. 提取 Target (101长度的最后一个点作为终极目标)
        nxt = chunk[-1][0]
        idx = self.max_len - 1

        # 4. 负采样过滤：必须用该用户的【全局】真实历史，绝不能只用当前 chunk！
        ts = set(map(lambda x: x[0], self.user_train[user_id]))

        # 5. 倒序装填 chunk[:-1] (前100个输入点)
        for i in reversed(chunk[:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0:
                # 随机生成一个不在 ts (全局历史) 里的负样本
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1:
                break

        # 6. 🔥 极致丝滑的 O(1) 提取：
        # 既然您假设外部已经按 sample_idx 算好并填好了矩阵，直接拿就完事了！
        user_time_matrix = self.time_matrix[sample_idx]
        user_dis_matrix = self.dis_matrix[sample_idx]

        # 7. 全副武装，转 Tensor 升天！
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(time_seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
            torch.tensor(user_time_matrix, dtype=torch.long),
            torch.tensor(user_dis_matrix, dtype=torch.long)
        )

    def __len__(self):
        # return len(self.user_train)
        return len(self.samples)


class Testdataset(Dataset):
    def __init__(self,all_u, all_seqs, all_time_matrix, all_distance_matrix, all_labels,user_poi_dict,itemnum,all_lens):
        self.all_u = all_u
        self.seqs = all_seqs
        self.time_matrix = all_time_matrix
        self.dis_matrix = all_distance_matrix
        self.labels = all_labels
        self.user_poi_dict = user_poi_dict
        self.n_poi =itemnum+1
        self.all_lens = all_lens

    def __getitem__(self, sample_idx):

        # 必须从 self.all_u 列表里，把这个题对应的【真实用户 ID】给揪出来！
        real_uid = self.all_u[sample_idx]

        # 2. 极其精准的切片提取
        seq = self.seqs[sample_idx]
        time_matrix = self.time_matrix[sample_idx]
        dis_matrix = self.dis_matrix[sample_idx]
        label = self.labels[sample_idx]
        seq_lens = self.all_lens[sample_idx]

        # 推荐系统测评极其残酷的规矩：用户去过的地方，如果不是这次的目标，统统打入冷宫（设为 1）！
        exclude_set = torch.LongTensor(list(set(self.user_poi_dict[real_uid])))
        exclude_mask = torch.zeros((self.n_poi,)).bool()
        exclude_mask[exclude_set] = 1

        # 绝对不能把它屏蔽！必须强行把目标 label 的 mask 撕开（设为 0）！
        exclude_mask[label] = 0

        # 4. 全副武装转 Tensor：注意第一个返回的必须是真实的 real_uid！
        uid_tensor = torch.tensor(real_uid, dtype=torch.long)
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        time_matrix_tensor = torch.tensor(time_matrix, dtype=torch.long)
        dis_matrix_tensor = torch.tensor(dis_matrix, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        len_tensor = torch.tensor(seq_lens, dtype=torch.long)

        return uid_tensor, seq_tensor, time_matrix_tensor, dis_matrix_tensor, label_tensor, exclude_mask,len_tensor

    def __len__(self):
        # 现在的长度是几万个测试切片，绝对不是用户总数！
        return len(self.seqs)


    def __len__(self):
        return len(self.seqs)
