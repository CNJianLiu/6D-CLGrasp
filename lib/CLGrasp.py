import torch
import torch.nn as nn

from lib.pspnet import PSPNet
from lib.pointnet import Pointnet2MSG
from lib.adaptor import PriorAdaptor
from lib.trans import Transformer as Transformer_s
from lib.trans_hypothesis import Transformer

class CLGraspNet(nn.Module):
    def __init__(self, n_cat=6, nv_prior=1024, num_structure_points=128):
        super(CLGraspNet, self).__init__()
        self.n_cat = n_cat
        self.psp = PSPNet(bins=(1, 2, 3, 6), backend='resnet18')
        self.instance_color = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.instance_geometry = Pointnet2MSG(0)
        self.num_structure_points = num_structure_points

        conv1d_stpts_prob_modules = []
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.ReLU())
        conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=256, out_channels=self.num_structure_points, kernel_size=1))
        conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

        self.lowrank_projection = None
        self.instance_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

        self.category_local = Pointnet2MSG(0)

        self.prior_enricher = PriorAdaptor(emb_dims=64, n_heads=4)

        self.category_global = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
        )

        self.norm_1 = nn.LayerNorm(1024)
        self.norm_2 = nn.LayerNorm(1024)
        self.trans_auto_1 = Transformer_s(depth=4, embed_dim=1024, mlp_hidden_dim=2048, h=8, drop_rate=0.1, length=1)
        self.trans_auto_2 = Transformer_s(depth=4, embed_dim=1024, mlp_hidden_dim=2048, h=8, drop_rate=0.1, length=1)
        self.Transformer = Transformer(3, 128*3, 1024, length=1024)
        
        self.assignment = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*nv_prior, 1), # n_cat*nv_prior = 6*1024
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, n_cat*3, 1), # n_cat*3 = 18
        )
        self.deformation[4].weight.data.normal_(0, 0.0001)

    def get_prior_enricher_lowrank_projection(self):
        return self.prior_enricher.get_lowrank_projection()
    
    def forward(self, points, img, choose, cat_id, prior):
        input_points = points.clone() # bs x 1024 x 3
        bs, n_pts = points.size()[:2] # bs, n_pts = bs, 1024
        nv = prior.size()[1] #1024
        points = self.instance_geometry(points) # bs x 64 x 1024, pointnet++
        index = cat_id + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
        # img = bs x 3 x 192 x 192
        out_img = self.psp(img) # bs x 32 x 192 x 192, PSPnet
        di = out_img.size()[1]
        emb = out_img.view(bs, di, -1)
        choose = choose.unsqueeze(1).repeat(1, di, 1)  # bs x 1 x 1024 -> bs x 32 x 1024
        emb = torch.gather(emb, 2, choose).contiguous()
        emb = self.instance_color(emb) # bs x 64 x 1024

        inst_local = torch.cat((points, emb), dim=1)     # bs x 128 x n_pts

        self.lowrank_projection = self.conv1d_stpts_prob(inst_local) # bs x 256 x 1024
        weighted_xyz = torch.sum(self.lowrank_projection[:, :, :, None] * input_points[:, None, :, :], dim=2) # bs x 256 x 3

        weighted_points_features = torch.sum(self.lowrank_projection[:, None, :, :] * points[:, :, None, :], dim=3) # bs x 64 x 256
        weighted_img_features = torch.sum(self.lowrank_projection[:, None, :, :] * emb[:, :, None, :], dim=3) # bs x 64 x 256

        del emb, choose, di, out_img, points, input_points, self.lowrank_projection

        # category-specific features
        cat_points = self.category_local(prior)    # bs x 64 x n_pts
        cat_color = self.prior_enricher(cat_points, weighted_points_features, weighted_img_features) # bs x 64 x 1024
        cat_local = torch.cat((cat_points, cat_color), dim=1) # bs x 128 x 1024

        del weighted_points_features, weighted_img_features, cat_points, cat_color

        x_1 =  inst_local # bs x 128 x 1024 current
        x_3 = cat_local # bs x 128 x 1024 prior
        x_2 = x_1 - x_3 # bs x 128 x 1024 different
    
        x_1 = x_1.permute(0, 2, 1).contiguous() # bs x 1024 x 128
        x_2 = x_2.permute(0, 2, 1).contiguous() # bs x 1024 x 128
        x_3 = x_3.permute(0, 2, 1).contiguous() # bs x 1024 x 128
       
        x_1, x_2, x_3 = self.Transformer(x_1, x_2, x_3) # bs x 1024 x 128

        x_1 = x_1.permute(0, 2, 1).contiguous() # bs x 128 x 1024
        x_2 = x_2.permute(0, 2, 1).contiguous() # bs x 128 x 1024
        x_3 = x_3.permute(0, 2, 1).contiguous() # bs x 128 x 1024
        x_1 = self.pool(x_1) # bs x 128 x 1
        x_2 = self.pool(x_2) # bs x 128 x 1
        x_3 = self.pool(x_3) # bs x 128 x 1

        # assignemnt matrix
        assign_feat = torch.cat((inst_local, x_1.repeat(1, 1, n_pts), x_2.repeat(1, 1, n_pts), x_3.repeat(1, 1, n_pts)), dim=1)     # bs x 512 x n_pts
        assign_mat = self.assignment(assign_feat) # bs x (6*1024) x n_pts
        assign_mat = assign_mat.view(-1, nv, n_pts).contiguous()   # bs, nc*nv, n_pts ->f bs*nc, nv, n_pts (bs*6 x 1024 x n_pts) 
        assign_mat = torch.index_select(assign_mat, 0, index)   # bs x nv x n_pts  
        assign_mat = assign_mat.permute(0, 2, 1).contiguous()    # bs x n_pts x nv 

        # deformation field
        deform_feat = torch.cat((cat_local, x_3.repeat(1, 1, nv), x_2.repeat(1, 1, n_pts), x_1.repeat(1, 1, nv)), dim=1)       # bs x 512 x n_pts
        deltas = self.deformation(deform_feat) # bs x (6*3) x n_pts
        deltas = deltas.view(-1, 3, nv).contiguous()   # bs, nc*3, nv -> bs*nc, 3, nv  (bs*6 x 3 x n_pts)
        deltas = torch.index_select(deltas, 0, index)   # bs x 3 x 1024
        deltas = deltas.permute(0, 2, 1).contiguous()   # bs x 1024 x 3

        return weighted_xyz, assign_mat, deltas

