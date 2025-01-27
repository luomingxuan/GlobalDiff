import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        # 这里需要考虑id为0以及pading token的情况，并且item id从1开始算，所以+2了
        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        """
            param:
                log_seqs : sid seqs B*S
            return :
                log_feats :(B,S,D)
        """
  
        # (B,S,D)
        # seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        # 出现 seqs中出现0的无效
        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        # S
        tl = seqs.shape[1] # time dim len for enforce causality
        # attention mask (S,S)
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        #(B,S,D)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training       
        """
        forward
            param:
                user_ids : B
                log_seqs : B*S
                pos_seqs : B*S
                neg_seqs : B*S
            return :
                pos_logits : B*S 与pos_seqs对应 ，对应的sid命中的概率
                neg_logits : B*S 与neg_seqs对应 ，对应的sid命中的概率


        """ 
        # (B,S,D)
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        # 算哈达玛乘积 ？(B,S)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        #(B,S)
        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        """
        模型推理
            param: 
                user_ids: (B)
                log_seqs : (B,S)
                item_indices : (B,K),代表每个样本有多少候选者
            return :
                logits : (B,K) ,每一个候选者的分数

        """
        # （B,S,D）
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        # 只要最后一个item的数据,(B,D)
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        # (B,K,D) ,K为推理的时候采样的候选者
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        # 算点积距离 (B,K)   (B,K,D) matmuul (B,D,1) -> (B,K,1) -> (B,K) 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)


    def predictAll(self, log_seqs ): # for inference
        """
        模型推理,全部候选者
            param: 
                log_seqs : (B,S)
            return :
                logits : (B,S,V) ,每一个候选者的分数
                log_feats : (B,S,D) ,一个item的hidden 表示

        """
        # （B,S,D）
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        B,S,D = log_feats.size()
        # (V,D) 
        item_embs = self.item_emb.weight
        # (B,V,D)
        item_embs = item_embs.unsqueeze(0).repeat(B,1,1)
        # (B,D,V)
        item_embs=torch.transpose(item_embs,1,2)
        # 算点积距离 (B,S,V)   (B,S,D) matmuul (B,D,V) -> (B,S,V)
        logits=torch.bmm(log_feats,item_embs)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        # 不要padding token
        logits = logits[:,:,:-1]
        return logits,log_feats