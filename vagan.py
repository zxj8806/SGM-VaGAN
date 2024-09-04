import torch
import torch.nn as nn
from snn_model.snn_layers import *
from snn_model.snn_prior import *
from snn_model.snn_posterior import *
import torch.nn.functional as F


class SNNencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = 0
        in_channels = 1  # batch size
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16
        self.k = 20
        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                       out_channels=h_dim,
                       kernel_size=3,
                       stride=2,
                       padding=1,
                       bias=True,
                       bn=tdBatchNorm(h_dim),
                       spike=LIFSpike(),
                       is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False

        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1] * 2,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())
        self.prior = PriorBernoulliSTBP(k=20)
        self.posterior = PosteriorBernoulliSTBP(k=20)

    def forward(self, x, scheduled=False):
        # print("SNNencoder x.shape", x.shape)
        x = self.encoder(x)  # (N,C,H,W,T)
        # print("SNNencoder x_encoder.shape", x.shape)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # (N,C*H*W,T)
        # print("SNNencoder x_encoder_flatten.shape", x.shape)
        latent_x = self.before_latent_layer(x)  # (N,latent_dim,T)
        # print("SNNencoder latent_x.shape", latent_x.shape)
        sampled_z, q_z = self.posterior(latent_x)  # sampled_z:(B,C,1,1,T), q_z:(B,C,k,T)

        # print("SNNencoder sampled_z.shape, q_z.shape", sampled_z.shape, q_z.shape)
        p_z = self.prior(sampled_z, scheduled, self.p)
        return sampled_z, q_z, p_z

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p - init_p) * epoch / max_epoch + init_p


class SNNdecoder(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        # Build Decoder
        modules = []
        print("SNNdecoder tdLinear init (latent_dim + 10) (hidden_dims[-1] * 4)", latent_dim + 10, hidden_dims[-1] * 4)

        # self.decoder_input = tdLinear(latent_dim + 10,
        #                              hidden_dims[-1] * 4,
        #                              bias=True,
        #                              bn=tdBatchNorm(hidden_dims[-1] * 4),
        #                              spike=LIFSpike())

        self.decoder_input = tdLinear(latent_dim * 2,
                                      hidden_dims[-1] * 4,
                                      bias=True,
                                      bn=tdBatchNorm(hidden_dims[-1] * 4),
                                      spike=LIFSpike())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                tdConvTranspose(hidden_dims[i],
                                hidden_dims[i + 1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1,
                                bias=True,
                                bn=tdBatchNorm(hidden_dims[i + 1]),
                                spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            tdConvTranspose(hidden_dims[-1],
                            hidden_dims[-1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                            bias=True,
                            bn=tdBatchNorm(hidden_dims[-1]),
                            spike=LIFSpike()),
            tdConvTranspose(hidden_dims[-1],
                            out_channels=1,
                            # out_channels=3,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                            bn=None,
                            spike=None)
        )

        self.membrane_output_layer = MembraneOutputLayer()

    # def forward(self, z, cond):
    # print("SNNdecoder forward z.shape, cond.shape", z.shape, cond.shape)
    # z = torch.cat([z, cond], dim=1)
    # result = self.decoder_input(z)  # (N,C*H*W,T)
    # result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N,C,H,W,T)
    # result = self.decoder(result)  # (N,C,H,W,T)
    # result = self.final_layer(result)  # (N,C,H,W,T)
    # out = torch.tanh(self.membrane_output_layer(result))
    # return out
    def forward(self, z, cond):
        # print("SNNdecoder forward z.shape, cond.shape", z.shape, cond.shape)

        z = torch.cat([z, z], dim=1)  # Concatenate along the latent dimension

        # print("Adjusted z shape for concatenation:", z.shape)
        result = self.decoder_input(z)  # (N, C*H*W, T)
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps)  # (N, C, H, W, T)
        result = self.decoder(result)  # (N, C, H, W, T)
        result = self.final_layer(result)  # (N, C, H, W, T)
        out = torch.tanh(self.membrane_output_layer(result))
        return out


class SNNgenerator(nn.Module):
    def __init__(self):
        super().__init__()

        latent_dim = 128
        self.latent_dim = latent_dim
        self.n_steps = 16

        self.k = 20

        hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims.copy()

        self.fsencoder = SNNencoder()
        self.fsdecoder = SNNdecoder()
        self.psp = PSP()

    def forward(self, x, cond, scheduled=False):
        # print("SNNgenerator forward x.shape, cond.shape", x.shape, cond.shape)#[64, 1, 32, 32] [64]

        if x.dim() == 4:
            x = x.unsqueeze(2)  # 现在x的形状将变为[64, 1, 1, 32, 32]
        # print("SNNgenerator forward x.shape", x.shape)#[64, 1, 32, 32] [64]

        sampled_z, q_z, p_z = self.fsencoder(x, scheduled)
        # print("sampled_z.shape, q_z.shape, p_z.shape", sampled_z.shape, q_z.shape, p_z.shape)#[1, 128, 16], [1, 128, 20, 16], [1, 128, 20, 16]
        x_recon = self.fsdecoder(sampled_z, cond)
        # print("x_recon.shape", x_recon.shape)#[1, 3, 32, 32]
        return x_recon, q_z, p_z, sampled_z

    def sample(self, batch_size=64):  # cond sample
        choice = torch.randint(low=0, high=10, size=(1, batch_size)).cuda()
        label = torch.nn.functional.one_hot(choice, 10)
        label = label.reshape([label.shape[1], 10, 1])
        label = label.repeat(1, 1, 16)
        sampled_z = self.fsencoder.prior.sample(batch_size)
        sampled_x = self.fsdecoder(sampled_z, label)
        return sampled_x, sampled_z

    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2)  # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2)  # (N, latent_dim, T)
        mmd_loss = torch.mean((self.psp(q_z_ber) - self.psp(p_z_ber)) ** 2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Distance_Loss': mmd_loss}

    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4, 4)


UP_MODES = ['nearest', 'bilinear']
NORMS = ['in', 'bn']

gf_dim = 256
df_dim = 128
g_spectral_norm = False
d_spectral_norm = True
bottom_width = 4


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)

        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        # print("OptimizedDisBlock residual input shape", x.shape)
        # 检查输入是否为单通道，如果是，则复制到三通道
        # if x.shape[1] == 1:
        #    x = x.repeat(1, 3, 1, 1)  # 在通道维度复制

        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        # if x.shape[1] == 1:
        #    x = x.repeat(1, 3, 1, 1)  # 在通道维度复制
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

    # def residual(self, x):
    #    h = x
    #    print("OptimizedDisBlock residual h.shape", h.shape)
    #    h = self.c1(h)
    #    h = self.activation(h)
    #    h = self.c2(h)
    #    h = _downsample(h)
    #    return h

    # def shortcut(self, x):
    #    return self.c_sc(_downsample(x))

    # def forward(self, x):
    #    return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DualDiscriminator(nn.Module):
    def __init__(self, in_channels=1, cont_dim=16, activation=nn.ReLU()):  # Changed in_channels to 3 for RGB
        super(DualDiscriminator, self).__init__()
        self.ch = df_dim  # Assuming df_dim = 128
        self.activation = activation
        self.block1 = OptimizedDisBlock(in_channels, self.ch)  # This will now accept 3-channel inputs
        self.block2 = DisBlock(self.ch, self.ch * 2, activation=activation, downsample=True)
        self.block3 = DisBlock(self.ch * 2, self.ch * 4, activation=activation, downsample=False)
        self.block4 = DisBlock(self.ch * 4, self.ch * 8, activation=activation, downsample=False)
        self.head_disc = nn.utils.spectral_norm(nn.Linear(self.ch * 8, 1))
        self.l5 = nn.utils.spectral_norm(nn.Linear(self.ch * 8, cont_dim, bias=False))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head_b1 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(256, cont_dim, bias=False)
        )
        self.head_b2 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(64, cont_dim, bias=False)
        )
        self.head_b3 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(64, cont_dim, bias=False)
        )
        self.head_b4 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear(64, cont_dim, bias=False)
        )

    def forward(self, x, mode="dual"):
        # print("DualDiscriminator forward x.shape", x.shape)
        h = x
        h1 = self.block1(h)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h = self.global_pool(h4)  # Pool to 1x1
        h = h.view(h.size(0), -1)  # Flatten to [batch_size, features]
        # print("Reshaped h.shape", h.shape)
        disc_out = self.head_disc(h)

        if mode == "dis":
            return disc_out
        elif mode == "cont":
            cont_out = {
                "b1-raw": h1,
                "b2-raw": h2,
                "b3-raw": h3,
                "b4-raw": h4,
                "b1": self.head_b1(h1),
                "b2": self.head_b2(h2),
                "b3": self.head_b3(h3),
                "b4": self.head_b4(h4),
                "final": h
            }
            return cont_out
        elif mode == "cont_local":
            cont_out = {
                "local_h1": h1,  # 128x16x16
                "local_h2": h2,  # 128x8x8
                "local_h3": h3,  # 128x8x8
                "local_h4": h4,  # 128x8x8
                "b1": self.head_b1(h1),
                "final": h
            }
            return cont_out
        return disc_out
