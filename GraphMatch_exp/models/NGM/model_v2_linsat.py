import itertools
import time

import torch
import torch.multiprocessing as mp
import numpy as np
import scipy.sparse as sp
from torch_sparse import spmm, SparseTensor

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from src.feature_align import feature_align
from src.factorize_graph_matching import construct_aff_mat, construct_sparse_aff_mat
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer, SPGNNLayer, PYGNNLayer
from LinSATNet import linsat_layer
import qpth
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from dense_apdagd_layer import dense_apdagd, DenseAPDAGDFunction
from dense_block_diag_apdagd_layer import dense_block_diag_apdagd, DenseBlockDiagAPDAGDFunction
from sparse_block_diag_apdagd_layer import sparse_csr_block_diag_from_tuple_list, sparse_block_diag_apdagd, \
    SparseBlockDiagAPDAGDFunction
from models.AFAT.sinkhorn_topk import greedy_perm
from src.lap_solvers.hungarian import hungarian

from src.utils.config import cfg

from src.backbone import *

CNN = eval(cfg.BACKBONE)


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def lexico_iter(lex):
    return itertools.combinations(lex, 2)


def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms


def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1)


def cvxpylayers_project(A, b, c, u, temp):
    x_cp = cp.Variable(A.shape[1], nonneg=True)
    c_cp = cp.Parameter(A.shape[1])
    objective = cp.Minimize(cp.sum(cp.multiply(c_cp, x_cp)
                                   - temp * cp.entr(cp.multiply(1. / u, x_cp))
                                   - temp * cp.entr(1. - cp.multiply(1. / u, x_cp))))
    constraints = [A @ x_cp == b, x_cp >= 0, x_cp <= u]
    prob = cp.Problem(objective, constraints)
    opt_layer = CvxpyLayer(prob, parameters=[c_cp], variables=[x_cp])
    x, = opt_layer(c)
    return x


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.sparse = cfg.NGM.SPARSE_MODEL
        self.gnn_layer = cfg.NGM.GNN_LAYER

        self.project_temp = cfg.PROJECT_TEMP
        self.project_max_iter = cfg.PROJECT_MAX_ITER
        self.project_way = cfg.PROJECT_WAY
        if cfg.PROJECT_DTYPE == 'float32':
            self.project_dtype = torch.float32
        elif cfg.PROJECT_DTYPE == 'float64':
            self.project_dtype = torch.float64
        else:
            raise ValueError(f"Undefined project_dtype: {cfg.PROJECT_DTYPE}")

        if not self.sparse:
            for i in range(self.gnn_layer):
                tau = cfg.NGM.SK_TAU
                if i == 0:
                    gnn_layer = GNNLayer(1, 1,
                                         cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                         sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                else:
                    gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                         cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                         sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                self.add_module('gnn_layer_{}'.format(i), gnn_layer)
        else:
            self.geometric = True
            if self.geometric:
                for i in range(self.gnn_layer):
                    tau = cfg.NGM.SK_TAU
                    if i == 0:
                        gnn_layer = PYGNNLayer(1, 1,
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    else:
                        gnn_layer = PYGNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            else:
                for i in range(self.gnn_layer):
                    tau = cfg.NGM.SK_TAU
                    if i == 0:
                        gnn_layer = SPGNNLayer(1, 1,
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    else:
                        gnn_layer = SPGNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                               cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                               sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=cfg.NGM.EDGE_EMB)
                    self.add_module('gnn_layer_{}'.format(i), gnn_layer)

        self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)

    def forward(
        self,
        data_dict,
    ):
        images = data_dict['images']
        points = data_dict['Ps']
        n_points = data_dict['ns']
        A_src, A_tgt = data_dict['As']
        graphs = data_dict['pyg_graphs']
        batch_size = data_dict['gt_perm_mat'].shape[0]
        num_graphs = len(images)

        global_list = []
        orig_graph_list = []
        node_feature_list = []
        for image, p, n_p, graph in zip(images, points, n_points, graphs):
            # extract feature
            nodes = self.node_layers(image)
            edges = self.edge_layers(nodes)

            global_list.append(self.final_layers(edges).reshape((nodes.shape[0], -1)))
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            U = concat_features(feature_align(nodes, p, n_p, self.rescale), n_p)
            F = concat_features(feature_align(edges, p, n_p, self.rescale), n_p)
            node_features = torch.cat((U, F), dim=1)
            node_feature_list.append(node_features.detach())
            graph.x = node_features

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        unary_affs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        quadratic_affs_list = [[0.5 * x for x in quadratic_affs] for quadratic_affs in quadratic_affs_list]

        s_list, mgm_s_list, x_list, mgm_x_list, indices = [], [], [], [], []

        for unary_affs, quadratic_affs, (idx1, idx2) in zip(unary_affs_list, quadratic_affs_list, lexico_iter(range(num_graphs))):
            if not self.sparse:
                kro_G, kro_H = data_dict['KGHs'] if num_graphs == 2 else data_dict['KGHs']['{},{}'.format(idx1, idx2)]
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K = construct_aff_mat(Ke, Kp, kro_G, kro_H)
                if num_graphs == 2: data_dict['aff_mat'] = K

                if cfg.NGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(K.shape[0], K.shape[1], 1, device=K.device)

                if cfg.NGM.POSITIVE_EDGES:
                    A = (K > 0).to(K.dtype)
                else:
                    A = (K != 0).to(K.dtype)

                emb_K = K.unsqueeze(-1)

                # NGM qap solver
                for i in range(self.gnn_layer):
                    gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                    emb_K, emb = gnn_layer(A, emb_K, emb, n_points[idx1], n_points[idx2])
            else:
                kro_G, kro_H = data_dict['KGHs_sparse'] if num_graphs == 2 else data_dict['KGHs_sparse']['{},{}'.format(idx1, idx2)]
                Kp = torch.stack(pad_tensor(unary_affs), dim=0)
                Ke = torch.stack(pad_tensor(quadratic_affs), dim=0)
                K_value, row_idx, col_idx = construct_sparse_aff_mat(Ke, Kp, kro_G, kro_H)

                if cfg.NGM.FIRST_ORDER:
                    emb = Kp.transpose(1, 2).contiguous().view(Kp.shape[0], -1, 1)
                else:
                    emb = torch.ones(cfg.BATCH_SIZE, Kp.shape[1] * Kp.shape[2], 1, device=K_value.device)

                # NGM qap solver
                if self.geometric:
                    adj = SparseTensor(row=row_idx.long(), col=col_idx.long(), value=K_value,
                                       sparse_sizes=(Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2]))
                    for i in range(self.gnn_layer):
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb = gnn_layer(adj, emb, n_points[idx1], n_points[idx2])
                else:
                    K_index = torch.cat((row_idx.unsqueeze(0), col_idx.unsqueeze(0)), dim=0).long()
                    A_value = torch.ones(K_value.shape, device=K_value.device)
                    tmp = torch.ones([Kp.shape[1] * Kp.shape[2]], device=K_value.device).unsqueeze(-1)
                    normed_A_value = 1 / torch.flatten(
                        spmm(K_index, A_value, Kp.shape[1] * Kp.shape[2], Kp.shape[1] * Kp.shape[2], tmp))
                    A_index = torch.linspace(0, Kp.shape[1] * Kp.shape[2] - 1, Kp.shape[1] * Kp.shape[2]).unsqueeze(0)
                    A_index = torch.repeat_interleave(A_index, 2, dim=0).long().to(K_value.device)

                    for i in range(self.gnn_layer):
                        gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                        emb = gnn_layer(K_value, K_index, normed_A_value, A_index, emb, n_points[idx1], n_points[idx2])

            v = self.classifier(emb)
            s = v.view(v.shape[0], points[idx2].shape[1], -1).transpose(1, 2)

            gt_ks = torch.tensor(
                [torch.sum(data_dict['gt_perm_mat'][i]) for i in range(data_dict['gt_perm_mat'].shape[0])],
                dtype=torch.float32, device=s.device)

            torch.cuda.reset_peak_memory_stats(device=s.device)
            max_memory_before_project = torch.cuda.max_memory_allocated(device=s.device) / 1024 / 1024
            # max_memory_reserved_before_project = torch.cuda.max_memory_reserved(device=s.device) / 1024 / 1024
            p0s = [n_points[idx1][ii] for ii in range(s.shape[0])]
            p1s = [n_points[idx2][ii] for ii in range(s.shape[0])]
            if self.project_way == 'qpth':
                # Qs = torch.block_diag(
                #     *[self.project_temp * torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device)
                #       for ii in range(s.shape[0])])
                # ps = torch.cat([- s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype)
                #                 for ii in range(s.shape[0])])
                # Gs = torch.block_diag(*[torch.cat([
                #     torch.kron(torch.eye(p0s[ii], dtype=self.project_dtype, device=s.device),
                #                torch.ones((1, p1s[ii]), dtype=self.project_dtype, device=s.device)),
                #     torch.kron(torch.ones((1, p0s[ii]), dtype=self.project_dtype, device=s.device),
                #                torch.eye(p1s[ii], dtype=self.project_dtype, device=s.device)),
                #     torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                #     - torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                # ], dim=0) for ii in range(s.shape[0])])
                # hs = torch.cat([torch.cat([
                #     torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                #     torch.ones(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                #     torch.zeros(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                # ], dim=0) for ii in range(s.shape[0])])
                # As = torch.block_diag(*[torch.ones((1, p0s[ii] * p1s[ii]), dtype=self.project_dtype, device=s.device)
                #                         for ii in range(s.shape[0])])
                # bs = torch.cat([gt_ks[ii].reshape(1).to(dtype=self.project_dtype) for ii in range(s.shape[0])])
                #
                # st = time.time_ns()
                # outputs = qpth.qp.QPFunction(eps=1e-3, verbose=0, maxIter=100000)(Qs, ps, Gs, hs, As, bs)
                # outputs = outputs[0].to(torch.float32)
                # ed = time.time_ns()
                # print('project_time/s:', (ed - st) / 1e9)

                outputs_list = []
                project_time = 0.
                n_seg = s.shape[0]
                idx_list = np.split(np.arange(s.shape[0]), n_seg)
                for idx_i in range(len(idx_list)):
                    Qs = torch.block_diag(
                        *[self.project_temp * torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device)
                          for ii in idx_list[idx_i]])
                    ps = torch.cat([- s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype)
                                    for ii in idx_list[idx_i]])
                    Gs = torch.block_diag(*[torch.cat([
                        torch.kron(torch.eye(p0s[ii], dtype=self.project_dtype, device=s.device),
                                   torch.ones((1, p1s[ii]), dtype=self.project_dtype, device=s.device)),
                        torch.kron(torch.ones((1, p0s[ii]), dtype=self.project_dtype, device=s.device),
                                   torch.eye(p1s[ii], dtype=self.project_dtype, device=s.device)),
                        torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                        - torch.eye(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                    ], dim=0) for ii in idx_list[idx_i]])
                    hs = torch.cat([torch.cat([
                        torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                        torch.ones(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                        torch.zeros(p0s[ii] * p1s[ii], dtype=self.project_dtype, device=s.device),
                    ], dim=0) for ii in idx_list[idx_i]])
                    As = torch.block_diag(
                        *[torch.ones((1, p0s[ii] * p1s[ii]), dtype=self.project_dtype, device=s.device)
                          for ii in idx_list[idx_i]])
                    bs = torch.cat([gt_ks[ii].reshape(1).to(dtype=self.project_dtype)
                                    for ii in idx_list[idx_i]])

                    st = time.time_ns()
                    outputs_idx = qpth.qp.QPFunction(eps=1e-3, verbose=0, maxIter=100000)(
                        Qs, ps, Gs, hs, As, bs)
                    outputs_list.append(outputs_idx[0].to(torch.float32))
                    ed = time.time_ns()
                    project_time += (ed - st) / 1e9
                outputs = torch.cat(outputs_list)
                print('project_time/s:', project_time)

                outputs_split = torch.split(outputs, [p0s[ii] * p1s[ii] for ii in range(s.shape[0])])
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                if torch.any(outputs > 1.):
                    print(f'Warning: qpth output is greater than upper bound by {torch.max(outputs - 1.).item()}')
                if torch.any(outputs < 0.):
                    print(f'Warning: qpth output is smaller than lower bound by {torch.max(0. - outputs).item()}')
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = DifferentiableClamp.apply(
                        outputs_split[ii].reshape(p0s[ii], p1s[ii]),
                        0., 1.
                    )
            elif self.project_way == 'dense_block_diag_apdagd_direct' \
                    or self.project_way == 'dense_block_diag_apdagd_kkt':
                As = [torch.cat([
                    torch.cat([
                        torch.kron(torch.eye(p0s[ii], dtype=self.project_dtype, device=s.device),
                                   torch.ones((1, p1s[ii]), dtype=self.project_dtype, device=s.device)),
                        torch.kron(torch.ones((1, p0s[ii]), dtype=self.project_dtype, device=s.device),
                                   torch.eye(p1s[ii], dtype=self.project_dtype, device=s.device)),
                        torch.ones((1, p0s[ii] * p1s[ii]), dtype=self.project_dtype, device=s.device)
                    ], dim=0),
                    torch.eye(p0s[ii] + p1s[ii] + 1, p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                ], dim=1) for ii in range(s.shape[0])]
                bs = [torch.cat([
                    torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                    gt_ks[ii].reshape(1).to(dtype=self.project_dtype)
                ], dim=0) for ii in range(s.shape[0])]
                cs = [torch.cat([
                    - s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype),
                    torch.zeros(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device)
                ], dim=0) for ii in range(s.shape[0])]
                us = [torch.ones(p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device)
                      for ii in range(s.shape[0])]
                n_c = torch.tensor([p0s[ii] + p1s[ii] + 1 for ii in range(s.shape[0])],
                                   dtype=torch.int64, device=s.device)
                n_v = torch.tensor([p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii] for ii in range(s.shape[0])],
                                   dtype=torch.int64, device=s.device)

                st = time.time_ns()
                if self.project_way == 'dense_block_diag_apdagd_direct':
                    outputs, _ = dense_block_diag_apdagd(
                        A=torch.block_diag(*As), b=torch.cat(bs),
                        c=torch.cat(cs), u=torch.cat(us),
                        n_c=n_c, n_v=n_v, theta=1. / self.project_temp
                    )
                elif self.project_way == 'dense_block_diag_apdagd_kkt':
                    outputs, _ = DenseBlockDiagAPDAGDFunction.apply(
                        torch.block_diag(*As), torch.cat(bs),
                        torch.cat(cs), torch.cat(us),
                        n_c, n_v, 1. / self.project_temp
                    )
                    # outputs, _ = DenseAPDAGDFunction.apply(
                    #     torch.block_diag(*As).unsqueeze(0), torch.cat(bs).unsqueeze(0),
                    #     torch.cat(cs).unsqueeze(0), torch.cat(us).unsqueeze(0),
                    #     1. / self.project_temp
                    # )
                    # outputs = torch.squeeze(outputs, dim=0)
                else:
                    raise ValueError(f"Undefined project_way: {self.project_way}")
                outputs = outputs.to(dtype=torch.float32)
                ed = time.time_ns()
                print('project_time/s:', (ed - st) / 1e9)

                outputs_split = torch.split(outputs, n_v.tolist())
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = outputs_split[ii][0:(p0s[ii] * p1s[ii])].reshape(p0s[ii], p1s[ii])
            elif self.project_way == 'sparse_block_diag_apdagd_direct' \
                    or self.project_way == 'sparse_block_diag_apdagd_kkt':
                if self.project_dtype == torch.float16:
                    np_dtype = np.float16
                elif self.project_dtype == torch.float32:
                    np_dtype = np.float32
                elif self.project_dtype == torch.float64:
                    np_dtype = np.float64
                else:
                    raise ValueError(f"Undefined project_dtype: {self.project_dtype}")
                As = [sp.hstack([
                    sp.vstack([
                        sp.kron(sp.eye(p0s[ii], dtype=np_dtype), np.ones((1, p1s[ii]), dtype=np_dtype)),
                        sp.kron(np.ones((1, p0s[ii]), dtype=np_dtype), sp.eye(p1s[ii], dtype=np_dtype)),
                        np.ones((1, p0s[ii] * p1s[ii]), dtype=np_dtype)
                    ]),
                    sp.eye(p0s[ii] + p1s[ii] + 1, p0s[ii] + p1s[ii], dtype=np_dtype),
                ]) for ii in range(s.shape[0])]
                A: sp.csr_matrix = sp.block_diag(As, format='csr')
                bs = [torch.cat([
                    torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                    gt_ks[ii].reshape(1).to(dtype=self.project_dtype)
                ], dim=0) for ii in range(s.shape[0])]
                cs = [torch.cat([
                    - s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype),
                    torch.zeros(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device)
                ], dim=0) for ii in range(s.shape[0])]
                us = [torch.ones(p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device)
                      for ii in range(s.shape[0])]
                n_c = torch.tensor([p0s[ii] + p1s[ii] + 1 for ii in range(s.shape[0])],
                                   dtype=torch.int64, device=s.device)
                n_v = torch.tensor([p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii] for ii in range(s.shape[0])],
                                   dtype=torch.int64, device=s.device)

                st = time.time_ns()
                if self.project_way == 'sparse_block_diag_apdagd_direct':
                    outputs, _ = sparse_block_diag_apdagd(
                        A=torch.sparse_csr_tensor(torch.tensor(A.indptr, dtype=torch.int64),
                                                  torch.tensor(A.indices, dtype=torch.int64),
                                                  torch.tensor(A.data, dtype=self.project_dtype),
                                                  A.shape, device=s.device),
                        b=torch.cat(bs), c=torch.cat(cs), u=torch.cat(us),
                        n_c=n_c, n_v=n_v, theta=1. / self.project_temp
                    )
                elif self.project_way == 'sparse_block_diag_apdagd_kkt':
                    outputs, _ = SparseBlockDiagAPDAGDFunction.apply(
                        torch.sparse_csr_tensor(torch.tensor(A.indptr, dtype=torch.int64),
                                                torch.tensor(A.indices, dtype=torch.int64),
                                                torch.tensor(A.data, dtype=self.project_dtype),
                                                A.shape, device=s.device),
                        torch.cat(bs), torch.cat(cs), torch.cat(us),
                        n_c, n_v, 1. / self.project_temp
                    )
                else:
                    raise ValueError(f"Undefined project_way: {self.project_way}")
                outputs = outputs.to(dtype=torch.float32)
                ed = time.time_ns()
                print('project_time/s:', (ed - st) / 1e9)

                # torch.save(p0s, 'p0s.pt')
                # torch.save(p1s, 'p1s.pt')
                # torch.save(s, 's.pt')
                # torch.save(gt_ks, 'gt_ks.pt')
                # torch.save(data_dict['gt_perm_mat'], 'gt_perm_mat.pt')
                # torch.save(data_dict['ns'][0], 'src_ns.pt')
                # torch.save(data_dict['ns'][1], 'tgt_ns.pt')
                # print('save tensor finish')

                outputs_split = torch.split(outputs, n_v.tolist())
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = outputs_split[ii][0:(p0s[ii] * p1s[ii])].reshape(p0s[ii], p1s[ii])
            elif self.project_way == 'cvxpylayers':
                if self.project_dtype == torch.float16:
                    np_dtype = np.float16
                elif self.project_dtype == torch.float32:
                    np_dtype = np.float32
                elif self.project_dtype == torch.float64:
                    np_dtype = np.float64
                else:
                    raise ValueError(f"Undefined project_dtype: {self.project_dtype}")
                As = [sp.hstack([
                    sp.vstack([
                        sp.kron(sp.eye(p0s[ii], dtype=np_dtype), np.ones((1, p1s[ii]), dtype=np_dtype)),
                        sp.kron(np.ones((1, p0s[ii]), dtype=np_dtype), sp.eye(p1s[ii], dtype=np_dtype)),
                        np.ones((1, p0s[ii] * p1s[ii]), dtype=np_dtype)
                    ]),
                    sp.eye(p0s[ii] + p1s[ii] + 1, p0s[ii] + p1s[ii], dtype=np_dtype),
                ]) for ii in range(s.shape[0])]
                A: sp.csr_matrix = sp.block_diag(As, format='csr')
                bs = [torch.cat([
                    torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device),
                    gt_ks[ii].reshape(1).to(dtype=self.project_dtype)
                ], dim=0).cpu().detach().numpy() for ii in range(s.shape[0])]
                b = np.hstack(bs)
                cs = [torch.cat([
                    - s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype),
                    torch.zeros(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device)
                ], dim=0).share_memory_() for ii in range(s.shape[0])]
                c = torch.cat(cs)
                us = [np.ones(p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii], dtype=np_dtype)
                      for ii in range(s.shape[0])]
                u = np.hstack(us)

                # st = time.time_ns()
                # # pool = mp.Pool(processes=24)
                # # pool_args = []
                # # for ii in range(s.shape[0]):
                # #     pool_args.append((As[ii], bs[ii], cs[ii], us[ii], self.project_temp))
                # # result = pool.starmap_async(cvxpylayers_project, pool_args)
                # # pool.close()
                # # pool.join()
                # # outputs_split = [res.to(torch.float32) for res in result.get()]
                # x_cp = cp.Variable(A.shape[1], nonneg=True)
                # c_cp = cp.Parameter(A.shape[1])
                # objective = cp.Minimize(cp.sum(cp.multiply(c_cp, x_cp)
                #                                - self.project_temp * cp.entr(cp.multiply(1. / u, x_cp))
                #                                - self.project_temp * cp.entr(1. - cp.multiply(1. / u, x_cp))))
                # constraints = [A @ x_cp == b, x_cp >= 0, x_cp <= u]
                # prob = cp.Problem(objective, constraints)
                # opt_layer = CvxpyLayer(prob, parameters=[c_cp], variables=[x_cp])
                # # outputs, = opt_layer(c, solver_args={"solve_method": "ECOS", "abstol": 1e-3})
                # outputs, = opt_layer(c, solver_args={"solve_method": "SCS", "eps_abs": 1e-3})
                # outputs = outputs.to(dtype=torch.float32)
                # ed = time.time_ns()
                # print('project_time/s:', (ed - st) / 1e9)

                outputs_list = []
                st = time.time_ns()
                for ii in range(s.shape[0]):
                    x_cp = cp.Variable(As[ii].shape[1], nonneg=True)
                    c_cp = cp.Parameter(As[ii].shape[1])
                    objective = cp.Minimize(cp.sum(cp.multiply(c_cp, x_cp)
                                                   - self.project_temp * cp.entr(cp.multiply(1. / us[ii], x_cp))
                                                   - self.project_temp * cp.entr(1. - cp.multiply(1. / us[ii], x_cp))))
                    constraints = [As[ii] @ x_cp == bs[ii], x_cp >= 0, x_cp <= us[ii]]
                    prob = cp.Problem(objective, constraints)
                    opt_layer = CvxpyLayer(prob, parameters=[c_cp], variables=[x_cp])
                    # outputs, = opt_layer(cs[ii], solver_args={"solve_method": "ECOS", "abstol": 1e-3})
                    outputs, = opt_layer(cs[ii], solver_args={"solve_method": "SCS", "eps_abs": 1e-3})
                    outputs_list.append(outputs.to(dtype=torch.float32))
                outputs = torch.cat(outputs_list)
                ed = time.time_ns()
                print('project_time/s:', (ed - st) / 1e9)

                outputs_split = torch.split(outputs, [p0s[ii] * p1s[ii] + p0s[ii] + p1s[ii] for ii in range(s.shape[0])])
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                if torch.any(outputs > 1.):
                    print(f'Warning: cvxpylayers output is greater than upper bound by {torch.max(outputs - 1.).item()}')
                if torch.any(outputs < 0.):
                    print(f'Warning: cvxpylayers output is smaller than lower bound by {torch.max(0. - outputs).item()}')
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = DifferentiableClamp.apply(
                        outputs_split[ii][0:(p0s[ii] * p1s[ii])].reshape(p0s[ii], p1s[ii]),
                        0., 1.
                    )
            elif self.project_way == 'sparse_linsat':
                if self.project_dtype == torch.float16:
                    np_dtype = np.float16
                elif self.project_dtype == torch.float32:
                    np_dtype = np.float32
                elif self.project_dtype == torch.float64:
                    np_dtype = np.float64
                else:
                    raise ValueError(f"Undefined project_dtype: {self.project_dtype}")
                As = [sp.vstack([
                    sp.kron(sp.eye(p0s[ii], dtype=np_dtype), np.ones((1, p1s[ii]), dtype=np_dtype)),
                    sp.kron(np.ones((1, p0s[ii]), dtype=np_dtype), sp.eye(p1s[ii], dtype=np_dtype))
                ]).tocoo() for ii in range(s.shape[0])]
                bs = [torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device) for ii in range(s.shape[0])]
                Es = [sp.coo_matrix(np.ones((1, p0s[ii] * p1s[ii]), dtype=np_dtype)) for ii in range(s.shape[0])]
                fs = [gt_ks[ii].reshape(1).to(dtype=self.project_dtype) for ii in range(s.shape[0])]
                inputs = [s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype) for ii in range(s.shape[0])]
                A: sp.coo_matrix = sp.block_diag(As)
                E: sp.coo_matrix = sp.block_diag(Es)

                st = time.time_ns()
                outputs = linsat_layer(
                    torch.cat(inputs),
                    A=torch.sparse_coo_tensor(
                        torch.tensor(np.vstack([A.row, A.col]), dtype=torch.int64),
                        torch.tensor(A.data, dtype=self.project_dtype), A.shape, device=s.device).coalesce(),
                    b=torch.cat(bs),
                    E=torch.sparse_coo_tensor(
                        torch.tensor(np.vstack([E.row, E.col]), dtype=torch.int64),
                        torch.tensor(E.data, dtype=self.project_dtype), E.shape, device=s.device).coalesce(),
                    f=torch.cat(fs),
                    max_iter=self.project_max_iter, tau=self.project_temp)
                outputs = outputs.to(torch.float32)
                ed = time.time_ns()
                print('project_time/s:', (ed - st) / 1e9)

                outputs_split = torch.split(outputs, [p0s[ii] * p1s[ii] for ii in range(s.shape[0])])
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = outputs_split[ii].reshape(p0s[ii], p1s[ii])
            elif self.project_way == 'linsat':
                As = [torch.cat([
                    torch.kron(torch.eye(p0s[ii], dtype=self.project_dtype, device=s.device),
                               torch.ones((1, p1s[ii]), dtype=self.project_dtype, device=s.device)),
                    torch.kron(torch.ones((1, p0s[ii]), dtype=self.project_dtype, device=s.device),
                               torch.eye(p1s[ii], dtype=self.project_dtype, device=s.device))
                ], dim=0) for ii in range(s.shape[0])]
                bs = [torch.ones(p0s[ii] + p1s[ii], dtype=self.project_dtype, device=s.device) for ii in range(s.shape[0])]
                Es = [torch.ones((1, p0s[ii] * p1s[ii]), dtype=self.project_dtype, device=s.device) for ii in range(s.shape[0])]
                fs = [gt_ks[ii].reshape(1).to(dtype=self.project_dtype) for ii in range(s.shape[0])]
                inputs = [s[ii, 0:p0s[ii], 0:p1s[ii]].reshape(-1).to(dtype=self.project_dtype) for ii in range(s.shape[0])]

                st = time.time_ns()
                outputs = linsat_layer(torch.cat(inputs), A=torch.block_diag(*As), b=torch.cat(bs),
                                       E=torch.block_diag(*Es), f=torch.cat(fs),
                                       max_iter=self.project_max_iter, tau=self.project_temp)
                outputs = outputs.to(torch.float32)
                ed = time.time_ns()
                print('project_time/s:', (ed - st) / 1e9)

                outputs_split = torch.split(outputs, [p0s[ii] * p1s[ii] for ii in range(s.shape[0])])
                ss_out = torch.zeros(s.shape, dtype=torch.float32, device=s.device)
                for ii in range(s.shape[0]):
                    ss_out[ii, 0:p0s[ii], 0:p1s[ii]] = outputs_split[ii].reshape(p0s[ii], p1s[ii])

                # for ii in range(s.shape[0]):
                #     p0 = n_points[idx1][ii]
                #     p1 = n_points[idx2][ii]
                #     constraint = torch.zeros(p0 + p1, p0 * p1,  ### p0 + p1 + 1
                #                              dtype=torch.float32, device=s.device)
                #     b = torch.zeros(p0 + p1, dtype=torch.float32, device=s.device)  ### p0 + p1 + 1
                #
                #     for cons_id in range(p0 + p1):
                #         tmp = torch.zeros(p0, p1, dtype=torch.float32, device=s.device)
                #         if cons_id < p0:
                #             tmp[cons_id, 0:p1] = 1
                #         else:
                #             tmp[0:p0, cons_id - p0] = 1
                #         constraint[cons_id, :] = tmp.reshape(-1)
                #         b[cons_id] = 1
                #
                #     E = torch.ones(1, p0 * p1, dtype=torch.float32, device=s.device)
                #     f = torch.zeros(1, dtype=torch.float32, device=s.device)
                #     f[0] = gt_ks[ii]
                #
                #     print(p0, p1, gt_ks[ii])
                #
                #     ### tmp = torch.ones(p0, p1, dtype=torch.float32, device=s.device)
                #     ### constraint[-1, :] = tmp.reshape(-1)
                #     ### b[-1] = gt_ks[ii]
                #
                #     input = s[ii, 0:p0, 0:p1].reshape(-1)
                #     ss_out[ii, 0:p0, 0:p1] = linsat_layer(input, A=constraint, b=b, E=E, f=f, max_iter=2 * cfg.NGM.SK_ITER_NUM,
                #                                      tau=self.tau).reshape(p0, p1)
            else:
                raise ValueError(f"Undefined project_way: {self.project_way}")
            max_memory_after_project = torch.cuda.max_memory_allocated(device=s.device) / 1024 / 1024
            # max_memory_reserved_after_project = torch.cuda.max_memory_reserved(device=s.device) / 1024 / 1024
            print('max_memory_allocated before project/MB:', max_memory_before_project)
            print('max_memory_allocated after project/MB:', max_memory_after_project)
            print('max_memory_allocated during project/MB:', max_memory_after_project - max_memory_before_project)
            # print('max_memory_reserved before project/MB:', max_memory_reserved_before_project)
            # print('max_memory_reserved after project/MB:', max_memory_reserved_after_project)
            # print('max_memory_reserved during project/MB:', max_memory_reserved_after_project - max_memory_reserved_before_project)
            # print('gt_ks:', gt_ks)

            x = hungarian(ss_out, n_points[idx1], n_points[idx2])
            top_indices = torch.argsort(x.mul(ss_out).reshape(x.shape[0], -1), descending=True, dim=-1)
            x = torch.zeros(ss_out.shape, device=ss_out.device)
            x = greedy_perm(x, top_indices, gt_ks)
            s_list.append(ss_out)
            x_list.append(x)
            indices.append((idx1, idx2))

        if cfg.PROBLEM.TYPE == '2GM' or cfg.PROBLEM.TYPE == 'IMT':
            data_dict.update({
                'ds_mat': s_list[0],
                'perm_mat': x_list[0],
            })
        elif cfg.PROBLEM.TYPE == 'MGM':
            data_dict.update({
                'ds_mat_list': mgm_s_list,
                'perm_mat_list': mgm_x_list,
                'graph_indices': indices,
            })

        return data_dict
