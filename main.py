import argparse
import gc
import os
import time
import random
from datetime import datetime

from dataset import Traindataset
from models import DePOI
from utils import *
import torch.nn.functional as F


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tky')#nyc,tky,gowalla
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--maxlen', default=100, type=int)
parser.add_argument('--hidden_units', default=64, type=int, help='Embedding size.')#64
parser.add_argument('--num_blocks', default=1, type=int, help='Number of stacked attention layer.')
parser.add_argument('--num_epochs', default=30, type=int)
parser.add_argument('--output_epochs', default=5, type=int)
parser.add_argument('--num_heads', default=2, type=int, help='Self attention heads')
parser.add_argument('--tran_gcn_layer', default=1, type=int)
parser.add_argument('--geo_gcn_layer', default=1, type=int)
parser.add_argument('--geo_weight', default=1, type=float)
parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0001, type=float)
parser.add_argument('--device', default='cuda', type=str)#cpu
parser.add_argument('--device_id', default='1', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--laplacian', default=True, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)
parser.add_argument('--dis_span', default=256, type=int)
parser.add_argument('--tran_reg', default=1.0, type=float)
parser.add_argument('--geo_reg', default=1.0, type=float)
parser.add_argument('--kl_reg', default=1.0, type=float)
parser.add_argument('--cb_reg_loss', default=0.01, type=float, help='Strength of causal-bias disagreement reg')
parser.add_argument('--tg_reg_loss', default=0.01, type=float, help='Strength of trans-geo contrastive loss')
parser.add_argument('--anchor_num', default=1000, type=int)#1000
parser.add_argument('--use_amp', default=True, type=str2bool)#新加,用于梯度优化
parser.add_argument('--grad_clip', default=5.0, type=float)
parser.add_argument('--history_prior_weight', default=7.5, type=float)#
parser.add_argument('--history_prior_min', default=0.0, type=float)
parser.add_argument('--transition_prior_weight', default=0.2, type=float)#
parser.add_argument('--transition_prior_smoothing', default=1e-3, type=float)
parser.add_argument('--transition_tail_weight', default=0.0, type=float)
parser.add_argument('--transition_tail_topk', default=20, type=int)
parser.add_argument('--transition_tail_preserve_topk', default=5, type=int)

args = parser.parse_args()


def mask(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value#正常poi是正常概率，为0的设置为-1e16
    return update_adj


def build_transition_log_prior(user_train, itemnum, smoothing):#马尔科夫链
    counts = np.zeros((itemnum + 1, itemnum + 1), dtype=np.float32)
    for history in user_train.values():
        for src, dst in zip(history[:-1], history[1:]):
            counts[src[0], dst[0]] += 1.0

    global_counts = counts.sum(axis=0)
    global_counts[0] = 0.0
    if global_counts.sum() <= 0:
        global_counts[1:] = 1.0
    fallback = global_counts + smoothing
    fallback[0] = 0.0
    fallback = fallback / np.maximum(fallback.sum(), 1e-12)

    row_sum = counts.sum(axis=1, keepdims=True)
    transition_prob = (counts + smoothing) / (row_sum + smoothing * itemnum)
    empty_rows = (row_sum.squeeze(-1) == 0)
    transition_prob[empty_rows] = fallback
    transition_prob[:, 0] = 0.0
    transition_prob = transition_prob / np.maximum(transition_prob.sum(axis=1, keepdims=True), 1e-12)
    return torch.log(torch.from_numpy(transition_prob + 1e-12))


def build_transition_tail_prior(user_train, itemnum, topk):#未调用
    counts = np.zeros((itemnum + 1, itemnum + 1), dtype=np.float32)
    for history in user_train.values():
        for src, dst in zip(history[:-1], history[1:]):
            counts[src[0], dst[0]] += 1.0

    counts[:, 0] = 0.0
    tail_prior = np.zeros_like(counts, dtype=np.float32)
    if topk <= 0:
        return torch.from_numpy(tail_prior)

    topk = min(topk, itemnum)
    rank_scores = np.linspace(1.0, 0.2, topk, dtype=np.float32)
    for src in range(1, itemnum + 1):
        row = counts[src]
        if row.sum() <= 0:
            continue
        top_idx = np.argpartition(row, -topk)[-topk:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        valid = row[top_idx] > 0
        tail_prior[src, top_idx[valid]] = rank_scores[:valid.sum()]
    return torch.from_numpy(tail_prior)


if __name__ == '__main__':
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))

    setup_seed(args.seed)

    time_string = datetime.now().strftime("%m%d%H%M%S")
    log_path = os.path.join('log', args.dataset + '_' + time_string + '.log')

    # data partition
    dataset = data_partition('data/' + args.dataset + '/' + args.dataset + '.txt')
    [user_train, user_valid, user_test, usernum, itemnum, timenum, pop_freq, user_poi_dict] = dataset
    # user_poi_dict = {user: list(set([item[0] for item in records])) for user, records in user_train.items()}

    # anchor_set = set(anchor_idx.tolist())
    # user_poi_dict = {user: list(set(poi_list) | anchor_set) for user, poi_list in user_poi_dict.items()}
    # print(user_poi_dict)
    num_batch = len(user_train) // args.batch_size

    # load transition distribution matrix
    # npz_path = 'data/' + args.dataset + '/' + args.dataset + '_mat.npz'
    # tra_adj_matrix = sp.load_npz(npz_path)
    # tra_adj_matrix = tra_adj_matrix.todok()

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('Batch number: {}. Average sequence length: {:.2f}'.format(num_batch, cc / len(user_train)))
    print('After filtering, #POI: {}, #user: {}'.format(itemnum, usernum))

    with open(os.path.join('log', args.dataset + '_' + time_string + '.log'), 'a+') as f:
        for k, v in sorted(vars(args).items()):
            f.write('{}: {}\n'.format(k, v))

    # generate or load spatio-temporal interval matrix
    mat_filename = 'data/{}/relation_matrix_{}_{}_{}.pickle'.format(
        args.dataset, args.dataset, args.maxlen, args.time_span)
    if os.path.exists(mat_filename):
        relation_matrix = pickle.load(open(mat_filename, 'rb'))
    else:
        relation_matrix = time_interval_mat(user_train, args.maxlen, args.time_span)
        pickle.dump(relation_matrix, open(mat_filename, 'wb'))

    dis_mat_filename = 'data/{}/relation_dis_matrix_{}_{}_{}.pickle'.format(
        args.dataset, args.dataset, args.maxlen, args.dis_span)
    if os.path.exists(dis_mat_filename):
        dis_relation_matrix = pickle.load(open(dis_mat_filename, 'rb'))
    else:
        dis_relation_matrix = dist_interval_mat(user_train, args.maxlen, args.dis_span)
        pickle.dump(dis_relation_matrix, open(dis_mat_filename, 'wb'))

    print('Done with spatio-temporal interval matrix.')

    print('Done with transition and geography graph matrix.')

    # load data
    train_dataset = Traindataset(user_train, relation_matrix, dis_relation_matrix, itemnum, args.maxlen)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)
    # create model
    # model = DePOI(usernum, itemnum, tran_mat, dist_mat, args).to(args.device)
    model = DePOI(usernum, itemnum, args).to(args.device)
    transition_log_prior = build_transition_log_prior(user_train, itemnum, args.transition_prior_smoothing)
    model.set_transition_prior(transition_log_prior)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量 (Total): {total_params / 1e6:.2f} M")
    print(f"可训练参数量 (Trainable): {trainable_params / 1e6:.2f} M")
    print(f"参数占用比例: {100 * trainable_params / total_params:.2f}%")

    if args.transition_tail_weight > 0:
        transition_tail_prior = build_transition_tail_prior(user_train, itemnum, args.transition_tail_topk)
        model.set_transition_tail_prior(transition_tail_prior)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass  # just ignore those failed init layers

    model.train()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)

    ce_criterion = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    parameters = [{'params': weight_decay_list},
                  {'params': no_decay_list, 'weight_decay': 0.}]
    adam_optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)
    use_amp = args.use_amp and args.device.startswith('cuda') and torch.cuda.is_available()
    amp_device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print('AMP enabled: {}'.format(use_amp))
    # import bitsandbytes as bnb
    # # >>> 完美平替：仅仅是把 torch.optim.Adam 换成了 bnb.optim.Adam8bit <<<
    # adam_optimizer = bnb.optim.Adam8bit(parameters, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

    # # from torch.cuda.amp import autocast, GradScaler
    # from torch.amp import autocast, GradScaler
    # scaler = GradScaler('cuda')
    # 改成这行：
    # adam_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    T = 0.0
    t0 = time.time()
    anchor_num = args.anchor_num

    # 🔥 循环外部初始化：全 0 阵列等待刷新
    best_NDCG = [0.0, 0.0, 0.0]
    best_HR = [0.0, 0.0, 0.0]
    best_MRR = [0.0]  # 因为咱们之前 return 的是 [global_mrr / test_user_num]

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        print('Training on Epoch-{}'.format(epoch), end=' ')
        anchor_idx = torch.randperm(itemnum)[:anchor_num]#[3, 0]
        # tra_adj_matrix_anchor = tra_adj_matrix[anchor_idx.numpy(), :].todense()
        # prior = torch.FloatTensor(tra_adj_matrix_anchor).to(args.device)
        anchor_idx += 1


        if args.inference_only:
            break
        for step, instance in enumerate(dataloader):
            u, seq, time_seq, pos, neg, time_matrix, dis_matrix = instance
            # print(dis_matrix)
            adam_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                # (fin_logits, fin_logits_bias, tg_con_loss, cb_con_loss) = (model(u, seq, time_matrix, dis_matrix, pos, neg, anchor_idx))
                (fin_logits, fin_logits_bias, cb_con_loss) = (
                    model(u, seq, time_matrix, dis_matrix, pos, neg, anchor_idx))

                pos_label_for_crosse = pos.reshape(-1).to(args.device)#128*50压扁成6400
                indices_for_crosse = pos_label_for_crosse != 0#true，false
                pos_label_cross = pos_label_for_crosse[indices_for_crosse]#,例如2500个真实POI，那么保留2500，

                main_loss = ce_criterion(fin_logits[indices_for_crosse].float(), pos_label_cross.long())

                probabilities = torch.softmax(fin_logits_bias[indices_for_crosse].float(), dim=1)#2500*4000
                pop_freq_tensor = torch.as_tensor(pop_freq, dtype=probabilities.dtype, device=args.device).unsqueeze(0)#从(4000,)变成（1,4000）
                # scaled_pop_freq_tensor = pop_freq_tensor
                pop_distribution = pop_freq_tensor.expand(probabilities.shape[0], -1)#2500*4000
                probabilities = probabilities + 1e-9
                pop_distribution = torch.softmax(pop_distribution, dim=-1)
                bias_loss = kl_loss(torch.log(probabilities), pop_distribution).mean()

                # loss = (main_loss + cb_con_loss * args.cb_reg_loss + tg_con_loss * args.tg_reg_loss + bias_loss * args.kl_reg)
                loss = (main_loss + cb_con_loss * args.cb_reg_loss  + bias_loss * args.kl_reg)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(adam_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(adam_optimizer)
            scaler.update()

        print('loss: {:.4f}, bias loss: {:.4f}'.format(loss, bias_loss))
        # print('main_loss: {:.4f}, cb_con_loss: {:.4f}, tg_con_loss: {:.4f}'.format(main_loss, cb_con_loss,tg_con_loss))
        print('main_loss: {:.4f}, cb_con_loss: {:.4f}'.format(main_loss, cb_con_loss))

        # if epoch % args.output_epochs == 0:
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1

            before_test = time.time()
            # anchor_set = set(anchor_idx.tolist())
            # user_poi_dict = {user: list(set(poi_list) | anchor_set) for user, poi_list in user_poi_dict.items()}
            NDCG, HR, MRR = evaluate_test(model, dataset, args, user_poi_dict,itemnum)

            best_NDCG = [max(b, c) for b, c in zip(best_NDCG, NDCG)]
            best_HR = [max(b, c) for b, c in zip(best_HR, HR)]
            best_MRR = [max(b, c) for b, c in zip(best_MRR, MRR)]


            print(f'Evaluating on Epoch-{epoch}, time: {T:.2f}s, loss: {loss:.4f}')
            print(f'HitR@2 = {best_HR[0]:.4f}, HitR@5 = {best_HR[1]:.4f}, HitR@10 = {best_HR[2]:.4f}')
            print(f'NDCG@2 = {best_NDCG[0]:.4f}, NDCG@5 = {best_NDCG[1]:.4f}, NDCG@10 = {best_NDCG[2]:.4f}')
            print(f'MRR = {best_MRR[0]:.4f}')
            infer_time = time.time() - before_test
            print(f'Inference time: {infer_time:.2f}s')
            print('-' * 60)
            with open(os.path.join('log', args.dataset + '_' + time_string + '.log'), 'a+') as f:
                f.write(f'Epoch:{epoch + 1}, loss: {loss:.4f}\nHitR: {HR}, NDCG: {NDCG}, MRR: {MRR[0]:.4f}\n')
                f.write('-' * 60 + '\n')
            t0 = time.time()
            model.train()
        gc.collect()

    print("=====FIN=====")
