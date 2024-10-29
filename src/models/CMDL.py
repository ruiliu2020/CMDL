import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from common.critic_objectives import *

class CMDL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CMDL, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.mi_loss = config['mi_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.critic_hidden_dim = 1
        self.critic_layers = 1
        self.info_pos = config['info_pos']
        self.info_neg = config['info_neg']
        self.learning_rate = config['learning_rate']
        self.margin = config['margin']
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 3, bias=False)
        )

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
      
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_share_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.tau = 0.5
        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        ) 
  
        self.club_zx_zs = CLUBInfoNCECritic(self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')
        self.club_zy_zs = CLUBInfoNCECritic(self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')
        
        self.infonce_x_zs = InfoNCECritic(self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')
        self.infonce_y_zs = InfoNCECritic(self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')
        
        self.club_x_ys_cond = CLUBInfoNCECritic(self.embedding_dim + self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')
        self.club_y_xs_cond = CLUBInfoNCECritic(self.embedding_dim + self.embedding_dim, self.embedding_dim, self.critic_hidden_dim, self.critic_layers, activation='relu')

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))
        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings  

        self.image_feats_sim = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        self.text_feats_sim = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, self.image_feats_sim)
        self.image_embeds = torch.cat([image_user_embeds, self.image_feats_sim], dim=0)
        text_user_embeds = torch.sparse.mm(self.R, self.text_feats_sim)
        self.text_embeds = torch.cat([text_user_embeds, self.text_feats_sim], dim=0)
        
        att_common = torch.cat([self.query_common(self.image_embeds), self.query_common(self.text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * self.image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * self.text_embeds
        self.image_specific = self.image_embeds - common_embeds
        self.text_specific = self.text_embeds - common_embeds
        self.share = common_embeds
        
        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        share_prefer = self.gate_share_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, self.image_specific)
        sep_text_embeds = torch.multiply(text_prefer, self.text_specific)
        sep_share_embeds = torch.multiply(share_prefer, self.share)
        
        generated_weights = self.fc(content_embeds)
        weight = torch.tensor([generated_weights[:, 0].unsqueeze(dim=1).max(), generated_weights[:, 1].unsqueeze(dim=1).max(), generated_weights[:, 2].unsqueeze(dim=1).max()])
        normalized_weight = torch.nn.functional.softmax(weight, dim=0)
        side_embeds = normalized_weight[0] * sep_image_embeds +normalized_weight[1]  * sep_text_embeds + normalized_weight[2] * sep_share_embeds
        
        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)
        return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_users, content_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        
        m_batch_mf_loss, m_batch_emb_loss, m_batch_reg_loss = self.bpr_loss(side_embeds_users[users], side_embeds_items[pos_items],
                                                                      side_embeds_items[neg_items])
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_items[pos_items], self.info_pos) + self.InfoNCE(side_embeds_users[users], content_users[users], self.info_neg)

        high_zx_zs = self.club_zx_zs(self.text_specific[pos_items], self.share[pos_items]) + self.club_zx_zs(self.text_specific[neg_items], self.share[neg_items])
        high_zy_zs = self.club_zy_zs(self.image_specific[pos_items], self.share[pos_items]) + self.club_zy_zs(self.image_specific[neg_items], self.share[neg_items])
        
        low_x_zs = self.infonce_x_zs(self.text_embeds[pos_items], self.share[pos_items]) + self.infonce_x_zs(self.text_embeds[neg_items], self.share[neg_items])
        low_y_zs = self.infonce_y_zs(self.image_embeds[pos_items], self.share[pos_items]) + self.infonce_y_zs(self.image_embeds[neg_items], self.share[neg_items])
        
        cond_high_x_ys = self.club_x_ys_cond(torch.cat([self.share[pos_items], self.image_embeds[pos_items]], dim=1), self.text_embeds[pos_items]) + self.club_x_ys_cond(torch.cat([self.share[neg_items], self.image_embeds[neg_items]], dim=1), self.text_embeds[neg_items])
        cond_high_y_xs = self.club_y_xs_cond(torch.cat([self.share[pos_items], self.text_embeds[pos_items]], dim=1), self.image_embeds[pos_items]) + self.club_y_xs_cond(torch.cat([self.share[neg_items], self.text_embeds[neg_items]], dim=1), self.image_embeds[neg_items])
        
        self.total_mi_loss = high_zx_zs + high_zy_zs + low_x_zs + low_y_zs + cond_high_x_ys + cond_high_y_xs
        
        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.mi_loss * self.total_mi_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, _, _ = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
    
    def learning_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)
        
        low_zx_zs = self.club_zx_zs.learning_loss(self.text_specific[pos_items], self.share[pos_items]) + self.club_zx_zs.learning_loss(self.text_specific[neg_items], self.share[neg_items])
        low_zy_zs = self.club_zy_zs.learning_loss(self.image_specific[pos_items], self.share[pos_items]) + self.club_zy_zs.learning_loss(self.image_specific[neg_items], self.share[neg_items])
        cond_low_x_ys = self.club_x_ys_cond.learning_loss(torch.cat([self.share[pos_items], self.image_embeds[pos_items]], dim=1), self.text_embeds[pos_items]) + self.club_x_ys_cond.learning_loss(torch.cat([self.share[neg_items], self.image_embeds[neg_items]], dim=1), self.text_embeds[neg_items])
        cond_low_y_xs = self.club_y_xs_cond.learning_loss(torch.cat([self.share[pos_items], self.text_embeds[pos_items]], dim=1), self.image_embeds[pos_items]) + self.club_y_xs_cond.learning_loss(torch.cat([self.share[neg_items], self.text_embeds[neg_items]], dim=1), self.image_embeds[neg_items])
        
        learning_losses = [low_zx_zs, low_zy_zs, cond_low_x_ys, cond_low_y_xs]
    
        return sum(learning_losses)
    
    def get_optims(self):
        a = [self.parameters()]
        non_CLUB_params = [self.image_trs.parameters(),
                           self.fc.parameters(),
                           self.text_trs.parameters(), 
                           self.gate_image_prefer.parameters(), 
                           self.gate_text_prefer.parameters(), 
                           self.gate_share_prefer.parameters(), 
                           self.infonce_y_zs.parameters(), 
                           self.infonce_x_zs.parameters(), 
                           self.user_embedding.parameters(),
                           self.item_id_embedding.parameters(),
                           self.image_embedding.parameters(),
                           self.text_embedding.parameters()]
        CLUB_params = [self.club_zx_zs.parameters(), 
                       self.club_zy_zs.parameters(),
                       self.club_x_ys_cond.parameters(),
                       self.club_y_xs_cond.parameters()]

        non_CLUB_optims = [optim.AdamW(param, lr=self.learning_rate) for param in non_CLUB_params]
        CLUB_optims = [optim.AdamW(param, lr=self.learning_rate) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims, non_CLUB_params, CLUB_params
