import sys

import torch
import torch.nn as nn

from DeGCN import DeGCN

FLOAT_MIN = -sys.float_info.max


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))#128*64*50,通道问题
        outputs = outputs.transpose(-1, -2)#128*50*64
        outputs += inputs
        return outputs


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = nn.Linear(hidden_size, hidden_size)
        self.K_w = nn.Linear(hidden_size, hidden_size)
        self.V_w = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, dis_matrix_K, dis_matrix_V,
                abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        batch_size, seq_len, _ = Q.shape

        def split_heads_3d(x):
            return x.reshape(batch_size, seq_len, self.head_num, self.head_size).permute(0, 2, 1, 3).reshape(
                batch_size * self.head_num, seq_len, self.head_size)

        def split_heads_4d(x):
            return x.reshape(batch_size, seq_len, seq_len, self.head_num, self.head_size).permute(
                0, 3, 1, 2, 4).reshape(batch_size * self.head_num, seq_len, seq_len, self.head_size)

        Q_ = split_heads_3d(Q)
        K_ = split_heads_3d(K)
        V_ = split_heads_3d(V)

        time_matrix_K_ = split_heads_4d(time_matrix_K)
        time_matrix_V_ = split_heads_4d(time_matrix_V)
        dis_matrix_K_ = split_heads_4d(dis_matrix_K)
        dis_matrix_V_ = split_heads_4d(dis_matrix_V)
        abs_pos_K_ = split_heads_3d(abs_pos_K)
        abs_pos_V_ = split_heads_3d(abs_pos_V)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))#128*50*50
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += torch.einsum('bijd,bid->bij', time_matrix_K_, Q_)#128*50*50
        attn_weights += torch.einsum('bijd,bid->bij', dis_matrix_K_, Q_)

        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(1).expand(-1, self.head_num, -1).reshape(
            batch_size * self.head_num, seq_len).unsqueeze(-1)#128*50*1
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])#128*50*50
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)#128*50*50
        paddings = torch.full_like(attn_weights, torch.finfo(attn_weights.dtype).min)#128*50*50
        attn_weights = torch.where(time_mask, paddings, attn_weights)#条件，为真时，取a,为假时取b.128*50*50
        attn_weights = torch.where(attn_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += torch.einsum('bij,bijd->bid', attn_weights, time_matrix_V_)
        outputs += torch.einsum('bij,bijd->bid', attn_weights, dis_matrix_V_)#128*50*64

        outputs = outputs.reshape(batch_size, self.head_num, seq_len, self.head_size).permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.hidden_size)#128*50*64

        return outputs


class Contrastive_BPR(nn.Module):
    def __init__(self, beta=1):
        super(Contrastive_BPR, self).__init__()
        self.Activation = nn.Softplus(beta=beta)

    def forward(self, x, pos, neg):
        loss_logit = (x * neg).sum(-1) - (x * pos).sum(-1)
        return self.Activation(loss_logit).mean()


# main model
class DePOI(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(DePOI, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        # self.tran_mat = tran_mat
        # self.dist_mat = dist_mat
        self.device = args.device
        self.geo_weight = args.geo_weight
        self.history_prior_weight = getattr(args, 'history_prior_weight', 0.0)
        self.history_prior_min = getattr(args, 'history_prior_min', 1.0)
        self.transition_prior_weight = getattr(args, 'transition_prior_weight', 0.0)
        self.transition_tail_weight = getattr(args, 'transition_tail_weight', 0.0)
        self.transition_tail_preserve_topk = getattr(args, 'transition_tail_preserve_topk', 0)
        self.register_buffer('transition_log_prior', torch.empty(0), persistent=False)
        self.register_buffer('transition_tail_prior', torch.empty(0), persistent=False)

        self.tran_item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.geo_item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.item_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.causal_item_embs = None
        self.bias_item_embs = None

        self.tran_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.geo_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.causal_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)
        self.bias_proj = nn.Linear(args.hidden_units, args.hidden_units, bias=False)

        self.CL_builder = Contrastive_BPR()

        self.transition_gcn = DeGCN(input_dim=args.hidden_units,
                                    output_dim=args.hidden_units,
                                    node_num=self.item_num,
                                    anchor_num=args.anchor_num,
                                    layer=args.tran_gcn_layer,
                                    dropout=args.dropout_rate,
                                    device=args.device)

        self.geography_gcn = DeGCN(input_dim=args.hidden_units,
                                   output_dim=args.hidden_units,
                                   node_num=self.item_num,
                                   anchor_num=args.anchor_num,
                                   layer=args.geo_gcn_layer,
                                   dropout=args.dropout_rate,
                                   device=args.device)

        self.abs_pos_K_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.time_matrix_K_emb = nn.Embedding(args.time_span + 1, args.hidden_units)
        self.time_matrix_V_emb = nn.Embedding(args.time_span + 1, args.hidden_units)

        self.dis_matrix_K_emb = nn.Embedding(args.dis_span + 1, args.hidden_units)
        self.dis_matrix_V_emb = nn.Embedding(args.dis_span + 1, args.hidden_units)

        self.abs_pos_K_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = nn.Dropout(p=args.dropout_rate)

        self.dis_matrix_K_dropout = nn.Dropout(p=args.dropout_rate)
        self.dis_matrix_V_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.causal = nn.Linear(2 * args.hidden_units, args.hidden_units)
        self.dias= nn.Linear(2 * args.hidden_units, args.hidden_units)

        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units, args.num_heads,
                                                         args.dropout_rate, args.device)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def set_transition_prior(self, transition_log_prior):
        self.transition_log_prior = transition_log_prior.to(self.device)

    def set_transition_tail_prior(self, transition_tail_prior):
        self.transition_tail_prior = transition_tail_prior.to(self.device)

    def _add_history_prior(self, logits, log_seqs, causal_prefix=False):
        if self.history_prior_weight <= 0:
            return logits

        log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device)
        seq_len = log_seqs.shape[1]
        pos_weights = torch.linspace(self.history_prior_min, 1.0, steps=seq_len,
                                     dtype=logits.dtype, device=self.device)
        if causal_prefix:
            batch_size, seq_len = log_seqs.shape
            history = torch.zeros(batch_size, seq_len, self.item_num + 1,
                                  dtype=logits.dtype, device=self.device)
            history.scatter_(2, log_seqs.unsqueeze(-1), pos_weights.view(1, seq_len, 1).expand(batch_size, -1, -1))
            history[:, :, 0] = 0
            history = history.cummax(dim=1).values
            return logits + self.history_prior_weight * history.reshape(-1, self.item_num + 1)

        history = torch.zeros(log_seqs.shape[0], self.item_num + 1, dtype=logits.dtype, device=self.device)
        history.scatter_reduce_(1, log_seqs, pos_weights.view(1, seq_len).expand(log_seqs.shape[0], -1),
                                reduce='amax', include_self=True)
        history[:, 0] = 0
        return logits + self.history_prior_weight * history

    def _add_transition_prior(self, logits, source_items):
        if self.transition_prior_weight <= 0 or self.transition_log_prior.numel() == 0:
            return logits
        source_items = torch.as_tensor(source_items, dtype=torch.long, device=self.device)
        prior = self.transition_log_prior[source_items].to(dtype=logits.dtype)
        # 数学意义： 这是典型的贝叶斯定理在深度学习中的应用。后验概率 = 似然度 $\times$ 先验概率。
        # 在对数（Log）空间下，概率的乘法就优雅地变成了加法：
        # $\log(\text{Final}) = \log(\text{Neural}) + \lambda \times \log(\text{Prior})$。
        return logits + self.transition_prior_weight * prior

    def _add_transition_tail_prior(self, logits, source_items, log_seqs=None, causal_prefix=False):
        if self.transition_tail_weight <= 0 or self.transition_tail_prior.numel() == 0:
            return logits

        source_items = torch.as_tensor(source_items, dtype=torch.long, device=self.device)
        prior = self.transition_tail_prior[source_items].to(dtype=logits.dtype)

        if log_seqs is not None:
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device)
            if causal_prefix:
                batch_size, seq_len = log_seqs.shape
                history = torch.zeros(batch_size, seq_len, self.item_num + 1,
                                      dtype=torch.bool, device=self.device)
                history.scatter_(2, log_seqs.unsqueeze(-1), True)
                history[:, :, 0] = False
                history = history.cumsum(dim=1).bool().reshape(-1, self.item_num + 1)
            else:
                history = torch.zeros(log_seqs.shape[0], self.item_num + 1,
                                      dtype=torch.bool, device=self.device)
                history.scatter_(1, log_seqs, True)
                history[:, 0] = False
            prior = prior.masked_fill(history, 0)

        return logits + self.transition_tail_weight * prior

    def _rerank_transition_tail_prior(self, logits, source_items, log_seqs=None):
        if self.transition_tail_weight <= 0 or self.transition_tail_prior.numel() == 0:
            return logits

        preserve_topk = min(max(int(self.transition_tail_preserve_topk), 0), logits.shape[1] - 1)
        if preserve_topk <= 0:
            return self._add_transition_tail_prior(logits, source_items, log_seqs, causal_prefix=False)

        source_items = torch.as_tensor(source_items, dtype=torch.long, device=self.device)
        prior = self.transition_tail_prior[source_items].to(dtype=logits.dtype)

        if log_seqs is not None:
            log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device)
            history = torch.zeros(log_seqs.shape[0], self.item_num + 1,
                                  dtype=torch.bool, device=self.device)
            history.scatter_(1, log_seqs, True)
            history[:, 0] = False
            prior = prior.masked_fill(history, 0)

        top_values, top_indices = torch.topk(logits, k=preserve_topk, dim=1)
        keep_mask = torch.zeros_like(prior, dtype=torch.bool)
        keep_mask.scatter_(1, top_indices, True)
        prior = prior.masked_fill(keep_mask, 0)

        boosted = logits + self.transition_tail_weight * prior
        cap = top_values[:, -1].unsqueeze(1) - 1e-6
        boosted = torch.minimum(boosted, cap)
        reranked = torch.where(prior > 0, boosted, logits)
        reranked.scatter_(1, top_indices, logits.gather(1, top_indices))
        return reranked

    def _build_attention_context(self, log_seqs, time_matrices, dis_matrices):
        log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device)
        batch_size, seq_len = log_seqs.shape

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        abs_pos_K = self.abs_pos_K_emb_dropout(self.abs_pos_K_emb(positions))
        abs_pos_V = self.abs_pos_V_emb_dropout(self.abs_pos_V_emb(positions))

        time_matrices = torch.as_tensor(time_matrices, dtype=torch.long, device=self.device)#batch*L*L
        time_matrix_K = self.time_matrix_K_dropout(self.time_matrix_K_emb(time_matrices))
        time_matrix_V = self.time_matrix_V_dropout(self.time_matrix_V_emb(time_matrices))#batch*L*L*64

        dis_matrices = torch.as_tensor(dis_matrices, dtype=torch.long, device=self.device)
        dis_matrix_K = self.dis_matrix_K_dropout(self.dis_matrix_K_emb(dis_matrices))
        dis_matrix_V = self.dis_matrix_V_dropout(self.dis_matrix_V_emb(dis_matrices))

        timeline_mask = log_seqs == 0
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
        return (log_seqs, timeline_mask, attention_mask, time_matrix_K, time_matrix_V,
                dis_matrix_K, dis_matrix_V, abs_pos_K, abs_pos_V)

    def _seq2feats_with_context(self, context, item_embs):
        (log_seqs, timeline_mask, attention_mask, time_matrix_K, time_matrix_V,
         dis_matrix_K, dis_matrix_V, abs_pos_K, abs_pos_V) = context

        seqs = item_embs[log_seqs, :]#(128, 50, 64)
        seqs *= item_embs.shape[1] ** 0.5
        seqs = self.item_emb_dropout(seqs)#(128, 50, 64)
        seqs *= ~timeline_mask.unsqueeze(-1)#(128, 50, 64)，反转后，真的保存

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)#先归一化
            mha_outputs = self.attention_layers[i](Q, seqs,
                                                   timeline_mask, attention_mask,
                                                   time_matrix_K, time_matrix_V,
                                                   dis_matrix_K, dis_matrix_V,
                                                   abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)#128*50*64
            seqs = self.forward_layers[i](seqs)#128*50*64
            seqs *= ~timeline_mask.unsqueeze(-1)#128*50*64

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs):
        context = self._build_attention_context(log_seqs, time_matrices, dis_matrices)
        return self._seq2feats_with_context(context, item_embs)

    def forward(self, user_ids, log_seqs, time_matrices, dis_matrices, pos_seqs, neg_seqs, anchor_idx):
        anchor_idx = anchor_idx.to(self.device)
        self.anchor_idx = anchor_idx

        causal_tran_embs, bias_tran_embs= self.transition_gcn(self.tran_item_emb, self.anchor_idx)
        causal_geo_embs, bias_geo_embs = self.geography_gcn(self.geo_item_emb, self.anchor_idx)

        tran_pool = torch.mean(causal_tran_embs, dim=0, keepdim=True).repeat(causal_tran_embs.shape[0], 1)#第一步是[1,64]然后(5001, 64)
        geo_pool = torch.mean(causal_geo_embs, dim=0, keepdim=True).repeat(causal_geo_embs.shape[0], 1)
        tran_pool = self.tran_proj(tran_pool.to(self.device))
        geo_pool = self.geo_proj(geo_pool.to(self.device))
        # con_loss = (self.CL_builder(causal_tran_embs, tran_pool, geo_pool) +
        #             self.CL_builder(causal_geo_embs, geo_pool, tran_pool))#返回50001个数字张量

        # causal_item_embs = causal_tran_embs + causal_geo_embs * self.geo_weight
        causal_item_embs = torch.cat([causal_tran_embs, causal_geo_embs], dim=1)
        causal_item_embs = self.causal(causal_item_embs)
        # bias_item_embs = bias_tran_embs + bias_geo_embs * self.geo_weight
        bias_item_embs = torch.cat([bias_tran_embs, bias_geo_embs], dim=1)
        bias_item_embs = self.dias(bias_item_embs)
        # self.causal_item_embs = causal_item_embs
        # self.bias_item_embs = bias_item_embs

        causal_pool = torch.mean(causal_item_embs, dim=0, keepdim=True).repeat(causal_item_embs.shape[0], 1)
        bias_pool = torch.mean(bias_item_embs, dim=0, keepdim=True).repeat(bias_item_embs.shape[0], 1)
        norm_c_square = torch.sum((causal_item_embs - causal_pool) ** 2, dim=1)#（50001,1）
        norm_b_square = torch.sum((causal_item_embs - bias_pool) ** 2, dim=1)
        loss_per_item = (norm_c_square - norm_b_square) ** 2 + 1e-6
        # cb_con_loss = torch.sum(loss_per_item)
        # 【修改这里】：将所有的节点差异求均值，而不是求和！
        cb_con_loss = torch.mean(loss_per_item)

        context = self._build_attention_context(log_seqs, time_matrices, dis_matrices)
        log_feats = self._seq2feats_with_context(context, causal_item_embs)#128*50*64
        log_feats_bias = self._seq2feats_with_context(context, bias_item_embs )

        # pos_embs = causal_item_embs[torch.LongTensor(pos_seqs).to(self.device), :]#128*50*64
        # neg_embs = causal_item_embs[torch.LongTensor(neg_seqs).to(self.device), :]
        # pos_embs_bias = bias_item_embs[torch.LongTensor(pos_seqs).to(self.device), :]
        #
        # pos_logits = (log_feats * pos_embs).sum(dim=-1)#128*50,后边还有一个sum(dim=-1)
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # pos_logits_bias = (log_feats_bias * pos_embs_bias).sum(dim=-1)

        fin_logits = log_feats.matmul(causal_item_embs.transpose(0, 1))#128*50*4000
        fin_logits = fin_logits.reshape(-1, fin_logits.shape[-1])#变成128乘以50，,6400*4000
        flat_source_items = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device).reshape(-1)#把 [128, 50] 的log_seqs 展平为[6400]
        fin_logits = self._add_transition_prior(fin_logits, flat_source_items)#6400*4000
        fin_logits = self._add_history_prior(fin_logits, log_seqs, causal_prefix=True)
        fin_logits_bias = log_feats_bias.matmul(bias_item_embs.transpose(0, 1))#128*50*4000
        fin_logits_bias = fin_logits_bias.reshape(-1, fin_logits_bias.shape[-1])#变成128乘以50，,6400*4000

        # return (fin_logits, fin_logits_bias, con_loss, cb_con_loss)
        return (fin_logits, fin_logits_bias, cb_con_loss)

    def predict(self, user_ids, log_seqs, time_matrices, dis_matrices, seq_lens):
        poi_transition_emb, _= self.transition_gcn(self.tran_item_emb, self.anchor_idx)
        poi_geography_emb, _= self.geography_gcn(self.geo_item_emb, self.anchor_idx)
        # only use causal rep for prediction
        # item_embs = poi_transition_emb + poi_geography_emb * self.geo_weight
        causal_item_embs = torch.cat([poi_transition_emb, poi_geography_emb], dim=1)
        item_embs = self.causal(causal_item_embs)
        # print("item_embs:",len(item_embs))
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)

        # valid_seqs = [seq[-seq_len:] for seq, seq_len in zip(log_feats, seq_lens)]
        # final_feat = torch.stack([seq.mean(dim=0) for seq in valid_seqs], dim=0)

        final_feat = log_feats[:, -1, :]#128*64，抽取最后一步

        item_emb = item_embs

        logits = final_feat.matmul(item_emb.transpose(0, 1))
        log_seqs_tensor = torch.as_tensor(log_seqs, dtype=torch.long, device=self.device)
        logits = self._add_transition_prior(logits, log_seqs_tensor[:, -1])
        logits = self._add_history_prior(logits, log_seqs_tensor, causal_prefix=False)
        # logits = self._rerank_transition_tail_prior(logits, log_seqs_tensor[:, -1], log_seqs_tensor)

        return logits
