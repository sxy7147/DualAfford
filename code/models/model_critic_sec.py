import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        # print(pointcloud.shape)
        # print(xyz.shape)
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            # print(li_features.shape) # 1: 64 * 1024

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(Critic, self).__init__()

        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, inputs):
        input_net = torch.cat(inputs, dim=-1)
        hidden_net = F.leaky_relu(self.mlp1(input_net))
        net = self.mlp2(hidden_net)
        return net


class Network(nn.Module):
    def __init__(self, feat_dim, task_feat_dim, cp_feat_dim, dir_feat_dim, hidden_feat_dim=128, task_input_dim=3):
        super(Network, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.critic = Critic(input_dim=feat_dim * 2 + task_feat_dim + cp_feat_dim * 2 + dir_feat_dim * 2, hidden_dim=hidden_feat_dim)

        self.mlp_task = nn.Linear(task_input_dim, task_feat_dim)
        self.mlp_dir = nn.Linear(3 + 3, dir_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.BCELoss_withoutSigmoid = nn.BCELoss(reduction='none')

    def forward(self, pcs, task, cp1, cp2, dir1, dir2):
        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)   # B * feats * num_pts
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        task_feats = self.mlp_task(task)
        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        dir1_feats = self.mlp_dir(dir1)
        dir2_feats = self.mlp_dir(dir2)

        pred_result_logits = self.critic([task_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats])
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores

    def forward_n(self, pcs, task, cp1, cp2, dir1, dir2, rvs=100):
        batch_size = pcs.shape[0]
        pcs[:, 0] = cp1
        pcs[:, 1] = cp2
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]
        net2 = whole_feats[:, :, 1]

        task_feats = self.mlp_task(task)    # B * 3
        cp1_feats = self.mlp_cp(cp1)
        cp2_feats = self.mlp_cp(cp2)
        dir1_feats = self.mlp_dir(dir1)     # B * 6
        dir2_feats = self.mlp_dir(dir2)     # (B * rvs) * 6

        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_net2 = net2.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_cp2_feats = cp2_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_dir1_feats = dir1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)

        pred_result_logits = self.critic([expanded_task_feats, expanded_net1, expanded_net2,
                                          expanded_cp1_feats, expanded_cp2_feats, expanded_dir1_feats, dir2_feats])
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores


    def forward_n_finetune(self, pcs, task, cp1, cp2, dir1, dir2, rvs_ctpt=10, rvs=10):
        batch_size = pcs.shape[0]

        task_feats = self.mlp_task(task)    # B * dim
        cp1_feats = self.mlp_cp(cp1)        # (B * rvs_ctpt) * dim
        cp2_feats = self.mlp_cp(cp2)        # (B * rvs_ctpt) * dim
        dir1_feats = self.mlp_dir(dir1)     # (B * rvs_ctpt * rvs) * dim
        dir2_feats = self.mlp_dir(dir2)     # (B * rvs_ctpt * rvs) * dim

        cp1 = cp1.reshape(batch_size, rvs_ctpt, 3)
        cp2 = cp2.reshape(batch_size, rvs_ctpt, 3)
        pcs[:, 0: rvs_ctpt] = cp1  # B * N * 3
        pcs[:, rvs_ctpt: rvs_ctpt + rvs_ctpt] = cp2
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)   # B * feats_dim * num_pts
        net1 = whole_feats[:, :, 0: rvs_ctpt].permute(0, 2, 1).reshape(batch_size * rvs_ctpt, -1)  # (B * rvs_ctpt) * dim
        net2 = whole_feats[:, :, rvs_ctpt: rvs_ctpt + rvs_ctpt].permute(0, 2, 1).reshape(batch_size * rvs_ctpt, -1)

        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_net2 = net2.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1,  rvs_ctpt * rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp2_feats = cp2_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)

        pred_result_logits = self.critic([expanded_task_feats, expanded_net1, expanded_net2,
                                          expanded_cp1_feats, expanded_cp2_feats, dir1_feats, dir2_feats])     # (B * rvs_ctpt * rvs) * 1
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores



    def forward_n_diffCtpts(self, pcs, task, cp1, cp2, dir1, dir2, rvs_ctpt=10, rvs=10):    # topk = num_ctpt2
        batch_size = pcs.shape[0]

        pcs[:, 0] = cp1     # B * N * 3
        pcs = pcs.unsqueeze(dim=1).repeat(1, 1, rvs_ctpt, 1).reshape(batch_size * rvs_ctpt, -1, 3)      # (B * rvs_ctpt) * N * 3
        pcs[:, 1] = cp2     # (B * rvs_ctpt) * N * 3
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]         # (B * rvs_ctpt) * dim
        net2 = whole_feats[:, :, 1]

        task_feats = self.mlp_task(task)    # B * dim
        cp1_feats = self.mlp_cp(cp1)        # B * dim
        cp2_feats = self.mlp_cp(cp2)        # (B * rvs_ctpt) * dim
        dir1_feats = self.mlp_dir(dir1)     # (B * rvs_ctpt * rvs) * dim
        dir2_feats = self.mlp_dir(dir2)     # (B * rvs_ctpt * rvs) * dim

        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_net2 = net2.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1,  rvs_ctpt * rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1,  rvs_ctpt * rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp2_feats = cp2_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)

        pred_result_logits = self.critic([expanded_task_feats, expanded_net1, expanded_net2,
                                          expanded_cp1_feats, expanded_cp2_feats, dir1_feats, dir2_feats])     # (B * rvs_ctpt * rvs) * 1
        pred_scores = torch.sigmoid(pred_result_logits)

        return pred_scores


    def get_ce_loss_total(self, pred_logits, gt_labels):
        loss = self.BCELoss_withoutSigmoid(pred_logits, gt_labels.float())
        return loss
