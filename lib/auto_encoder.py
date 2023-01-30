import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=1):
				super(ConvBN, self).__init__(
						nn.Conv1d(C_in, C_out, kernel_size),
						nn.BatchNorm1d(C_out),
				)

class ConvBNReLU(nn.Sequential):
		def __init__(self, C_in, C_out, kernel_size=1):
				super(ConvBNReLU, self).__init__(
						nn.Conv1d(C_in, C_out, kernel_size),
						nn.BatchNorm1d(C_out),
						nn.ReLU(inplace=True)
				)

class ResnetBasicBlock(nn.Module):
	def __init__(self, inplanes, planes, kernel_size=1):
		super().__init__()
		self.conv1 = nn.Conv1d(inplanes, planes, kernel_size)
		self.bn1 = nn.BatchNorm1d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv1d(planes, planes, kernel_size)
		self.bn2 = nn.BatchNorm1d(planes)

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += identity
		out = self.relu(out)

		return out

class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, 1)
        self.conv2 = ResnetBasicBlock(64, 64)
        self.conv3 = ConvBNReLU(64, 128, 1)
        self.conv4 = ResnetBasicBlock(128, 128)
        self.conv5 = ConvBNReLU(256, 1024, 1)
        self.conv6 = ResnetBasicBlock(1024, 1024)
        self.fc = nn.Linear(1024, emb_dim)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)

        """
        np = xyz.size()[2]
        x = self.conv1(xyz)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        global_feat = F.adaptive_max_pool1d(x, 1)
        x = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
        embedding = self.fc(x)
        return embedding


class PointCloudDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3*n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)

        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, 3)
        return out_pc


class GSENet(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(GSENet, self).__init__()
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, n_pts)

    def forward(self, in_pc, emb=None):
        """
        Args:
            in_pc: (B, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)
            emb = self.encoder(xyz)
        out_pc = self.decoder(emb)
        return emb, out_pc
