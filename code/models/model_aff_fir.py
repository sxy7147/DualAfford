import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class ActionScore(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ActionScore, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    # feats B x F
    # output: B
    def forward(self, inputs):
        feats = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(feats))
        net = self.mlp2(net)
        return net


class Network(nn.Module):
    def __init__(self, feat_dim, task_feat_dim, cp_feat_dim, dir_feat_dim, topk=1, task_input_dim=3):
        super(Network, self).__init__()

        self.topk = topk

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.mlp_task = nn.Linear(task_input_dim, task_feat_dim)
        self.mlp_dir = nn.Linear(3 + 3, dir_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.action_score = ActionScore(feat_dim + task_feat_dim + cp_feat_dim)


    def forward(self, pcs, task, cp1):
        pcs[:, 0] = cp1
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        task_feats = self.mlp_task(task)
        cp1_feats = self.mlp_cp(cp1)

        pred_result_logits = self.action_score([net1, task_feats, cp1_feats])
        pred_score = torch.sigmoid(pred_result_logits)
        return pred_score

    def get_loss(self, pred_score, gt_score):
        loss = self.L1Loss(pred_score, gt_score).mean()
        return loss


    def inference_whole_pc(self, pcs, task):
        batch_size = pcs.shape[0]
        num_pts = pcs.shape[1]

        cp1 = pcs.view(batch_size * num_pts, -1)
        cp1_feats = self.mlp_cp(cp1)

        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats.permute(0, 2, 1).reshape(batch_size * num_pts, -1)

        task_feats = self.mlp_task(task)

        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1, num_pts, 1).reshape(batch_size * num_pts, -1)

        pred_result_logits = self.action_score([net1, expanded_task_feats, cp1_feats])
        pred_score = torch.sigmoid(pred_result_logits).reshape(batch_size, num_pts)
        return pred_score

