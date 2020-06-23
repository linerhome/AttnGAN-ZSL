import torch
import torch.nn as nn

from src.miscc.config import cfg

rdc_text_dim = cfg.ZSL.rdc_text_dim
z_dim = cfg.ZSL.z_dim
h_dim = cfg.ZSL.h_dim


class ZSLG(nn.Module):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 z_dim=z_dim,
                 h_dim=h_dim,
                 rdc_text_dim=rdc_text_dim,
                 mode='sent'
                 ):
        super().__init__()

        assert mode in ('sent', 'words')
        if mode == 'words':
            # TODO: Implement using word-level, matrix (not sentence-level, vector) features
            raise NotImplementedError

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.rdc_text_dim = rdc_text_dim

        self.rdc_text = nn.Linear(in_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, out_dim),
                                  nn.Tanh())  # TODO: Need to check what type of activation is needed, support for the output should be the same as for input

    def forward(self, c, z=None):
        if z is None:
            z = torch.randn((c.shape[0], self.z_dim), device=c.device)

        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output


class ZSLD(nn.Module):
    def __init__(self, num_classes=150, input_dim=256):
        super().__init__()
        self.D_shared = nn.Sequential(nn.Linear(input_dim, h_dim),
                                      nn.ReLU())
        self.D_gan = nn.Linear(h_dim, 1)
        self.D_aux = nn.Linear(h_dim, num_classes)

    def forward(self, input):
        h = self.D_shared(input)
        return self.D_gan(h), self.D_aux(h)


zsl_g = ZSLG
zsl_d = ZSLD
