import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class KFSSM(nn.Module):

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, N):
        dt = torch.exp(self.log_dt) # (L)
        C = torch.view_as_complex(self.C) # (L D)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (L D)
        B = A[:,0].unsqueeze(-1) * torch.arange(N, device=A.device)#(L N)
        B = B * dt.unsqueeze(-1)
        dtA = A * dt.unsqueeze(-1)  # (L D)
        K = dtA.unsqueeze(-1) * torch.arange(N, device=A.device) # (L D N)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        K2 = 2 * torch.einsum('hl, hl -> hl', K, B).real
        return K,K2

    def register(self, name, tensor, lr=None):

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model,self).__init__()

        self.h = configs.seq_len
        self.p = configs.pred_len
        self.n = configs.d_model
        self.channel = configs.c_out
        self.d_output = self.h
        self.pred_len = configs.pred_len
        self.D = nn.Parameter(torch.randn(self.h))
        self.log_variance = nn.Parameter(torch.log(torch.FloatTensor(1 + torch.rand(1))), requires_grad=True)
        self.kernel = KFSSM(self.h, N=self.n)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout1d(0.05)
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.p, kernel_size=1),
            nn.GLU(dim=1),
        )
        self.outporj = nn.Sequential(
        nn.Linear(self.h,2*self.p),
        nn.GLU(dim=2),
        )

    def forward(self, u, x_mark_enc, x_dec, x_mark_dec):
        """ Input and output shape (B, L, N) """
        
        x_mean = torch.mean(u, dim=1, keepdim=True)
        u = u - x_mean
        x_var=torch.var(u, dim=1, keepdim=True)+ 1e-5
        u = u / torch.sqrt(x_var)

        std = torch.sqrt(torch.exp(self.log_variance))
        mean = torch.zeros_like(u, device = u.device)
        W = torch.normal(mean, std.expand_as(mean))

        l = u.size(1)
        N = u.size(2)
        k,k2 = self.kernel(N=N) # (L N)
        
        # Convolution
        k_f = torch.fft.rfft(k.permute(1,0), n = 2*l) # (N L+1)
        k2_f = torch.fft.rfft(k2.permute(1,0), n=2*l) # (N L+1)
        w_f = torch.fft.rfft(W.permute(0,2,1), n=2*l) # (B N L+1)
        u_f = torch.fft.rfft(u.permute(0,2,1), n = 2*l) # (B N L+1)
        y = torch.fft.irfft(u_f*k2_f, n=2*l) # (B N L)
        xy = torch.fft.irfft(w_f*k_f, n=2*l) # (B N L)
        y = y + xy  # (B L N)
        y = y.permute(0,2,1)[:,-self.h:,:] + u * self.D.unsqueeze(-1)
        
        #y = self.output_linear(y.permute(0,2,1)).permute(0,2,1)
        y = self.dropout(self.outporj(y.permute(0,2,1)).permute(0,2,1))


        y= y * torch.sqrt(x_var) + x_mean
        return y[:,-self.pred_len:,:]
