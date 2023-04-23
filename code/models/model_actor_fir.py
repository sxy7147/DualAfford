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


class ActorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorEncoder, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)
        self.mlp3 = nn.Linear(output_dim, output_dim)
        self.get_mu = nn.Linear(output_dim, output_dim)
        self.get_logvar = nn.Linear(output_dim, output_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')


    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar


class ActorDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(ActorDecoder, self).__init__()

        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = self.mlp(net)
        return net


class Network(nn.Module):
    def __init__(self, feat_dim, task_feat_dim, cp_feat_dim, dir_feat_dim, z_dim=128, lbd_kl=1.0, lbd_dir=1.0, task_input_dim=3):
        super(Network, self).__init__()

        self.feat_dim = feat_dim
        self.z_dim = z_dim

        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir

        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})

        self.mlp_task = nn.Linear(task_input_dim, task_feat_dim)
        self.mlp_dir = nn.Linear(3 + 3, dir_feat_dim)
        self.mlp_cp = nn.Linear(3, cp_feat_dim)     # contact point

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.all_encoder = ActorEncoder(input_dim=feat_dim + task_feat_dim + cp_feat_dim + dir_feat_dim, output_dim=z_dim)
        self.decoder = ActorDecoder(input_dim=feat_dim + task_feat_dim + cp_feat_dim + z_dim, output_dim=6)


    def KL(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        # ipdb.set_trace()
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # high star implementation
        # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = torch.mean(loss)
        return loss

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        # [bug fixed]
        # pred_Rs = self.bgs(pred_6d.reshape(-1, 3, 2))
        # gt_Rs = self.bgs(gt_6d.reshape(-1, 3, 2))
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta


    def forward(self, pcs, task, cp1, dir1):
        pcs[:, 0] = cp1
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        task_feats = self.mlp_task(task)
        cp1_feats = self.mlp_cp(cp1)
        dir1_feats = self.mlp_dir(dir1)

        z_all, mu, logvar = self.all_encoder([net1, task_feats, cp1_feats, dir1_feats])
        recon_dir1 = self.decoder([net1, task_feats, cp1_feats, z_all])

        return recon_dir1, mu, logvar


    def get_loss(self, pcs, task, cp1, dir1):
        batch_size = pcs.shape[0]
        recon_dir1, mu, logvar = self.forward(pcs, task, cp1, dir1)
        dir_loss = self.get_6d_rot_loss(recon_dir1, dir1)
        dir_loss = dir_loss.mean()
        kl_loss = self.KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['dir'] = dir_loss
        losses['tot'] = self.lbd_kl * kl_loss + self.lbd_dir * dir_loss

        return losses


    def actor_sample(self, pcs, task, cp1):
        batch_size = task.shape[0]

        pcs[:, 0] = cp1
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        task_feats = self.mlp_task(task)
        cp1_feats = self.mlp_cp(cp1)

        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()

        recon_dir1 = self.decoder([net1, task_feats, cp1_feats, z_all])
        recon_dir1 = recon_dir1.reshape(-1, 2, 3).permute(0, 2, 1)
        recon_dir1 = self.bgs(recon_dir1)
        recon_dir1 = recon_dir1.permute(0, 2, 1)
        recon_dir1 = recon_dir1[:, :2, :]

        return recon_dir1


    def actor_sample_n(self, pcs, task, cp1, rvs=100):
        batch_size = pcs.shape[0]

        pcs[:, 0] = cp1
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0]

        task_feats = self.mlp_task(task)
        cp1_feats = self.mlp_cp(cp1)

        z_all = torch.Tensor(torch.randn(batch_size, rvs, self.z_dim)).to(net1.device)

        expanded_rvs = z_all.reshape(batch_size * rvs, -1)
        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs, -1)

        recon_dir1 = self.decoder([expanded_net1, expanded_task_feats, expanded_cp1_feats, expanded_rvs])
        recon_dir1 = recon_dir1.reshape(-1, 2, 3).permute(0, 2, 1)
        recon_dir1 = self.bgs(recon_dir1)
        recon_dir1 = recon_dir1.permute(0, 2, 1)
        recon_dir1 = recon_dir1[:, :2, :]

        return recon_dir1


    def actor_sample_n_finetune(self, pcs, task, cp1, rvs_ctpt=10, rvs=100):
        batch_size = pcs.shape[0]

        task_feats = self.mlp_task(task)    # (B, -1)
        cp1_feats = self.mlp_cp(cp1)        # (B * rvs_ctpt, -1)
        z_all = torch.Tensor(torch.randn(batch_size * rvs_ctpt, rvs, self.z_dim)).to(pcs.device)

        cp1 = cp1.reshape(batch_size, rvs_ctpt, 3)
        pcs[:, 0: rvs_ctpt] = cp1
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        net1 = whole_feats[:, :, 0: rvs_ctpt].permute(0, 2, 1).reshape(batch_size * rvs_ctpt, -1)

        expanded_rvs = z_all.reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_task_feats = task_feats.unsqueeze(dim=1).repeat(1, rvs_ctpt * rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_net1 = net1.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)
        expanded_cp1_feats = cp1_feats.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(batch_size * rvs_ctpt * rvs, -1)

        recon_dir1 = self.decoder([expanded_net1, expanded_task_feats, expanded_cp1_feats, expanded_rvs])
        recon_dir1 = self.bgs(recon_dir1.reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)

        return recon_dir1[:, :2, :]

