class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class LSKA3D(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 8:
            self.conv0d = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=dim)
            self.conv0h = nn.Conv3d(dim, dim, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), groups=dim)
            self.conv0v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), groups=dim)
            self.conv_dilated_d = nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0),
                                            groups=dim, dilation=2)
            self.conv_dilated_h = nn.Conv3d(dim, dim, kernel_size=(1, 4, 1), stride=(1, 1, 1), padding=(0, 3, 0),
                                            groups=dim, dilation=2)
            self.conv_dilated_v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 4), stride=(1, 1, 1), padding=(0, 0, 3),
                                            groups=dim, dilation=2)
        elif k_size == 16:
            self.conv0d = nn.Conv3d(dim, dim, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), groups=dim)
            self.conv0h = nn.Conv3d(dim, dim, kernel_size=(1, 7, 1), stride=(1, 1, 1), padding=(0, 3, 0), groups=dim)
            self.conv0v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 7), stride=(1, 1, 1), padding=(0, 0, 3), groups=dim)
            self.conv_dilated_d = nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(6, 0, 0),
                                            groups=dim, dilation=4)
            self.conv_dilated_h = nn.Conv3d(dim, dim, kernel_size=(1, 4, 1), stride=(1, 1, 1), padding=(0, 6, 0),
                                            groups=dim, dilation=4)
            self.conv_dilated_v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 4), stride=(1, 1, 1), padding=(0, 0, 6),
                                            groups=dim, dilation=4)

        elif k_size == 32:
            self.conv0d = nn.Conv3d(dim, dim, kernel_size=(15, 1, 1), stride=(1, 1, 1), padding=(7, 0, 0), groups=dim)
            self.conv0h = nn.Conv3d(dim, dim, kernel_size=(1, 15, 1), stride=(1, 1, 1), padding=(0, 7, 0), groups=dim)
            self.conv0v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 15), stride=(1, 1, 1), padding=(0, 0, 7), groups=dim)
            self.conv_dilated_d = nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(12, 0, 0),
                                            groups=dim, dilation=8)
            self.conv_dilated_h = nn.Conv3d(dim, dim, kernel_size=(1, 4, 1), stride=(1, 1, 1), padding=(0, 12, 0),
                                            groups=dim, dilation=8)
            self.conv_dilated_v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 4), stride=(1, 1, 1), padding=(0, 0, 12),
                                            groups=dim, dilation=8)
        elif k_size == 64:
            self.conv0d = nn.Conv3d(dim, dim, kernel_size=(15, 1, 1), stride=(1, 1, 1), padding=(7, 0, 0), groups=dim)
            self.conv0h = nn.Conv3d(dim, dim, kernel_size=(1, 15, 1), stride=(1, 1, 1), padding=(0, 7, 0), groups=dim)
            self.conv0v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 15), stride=(1, 1, 1), padding=(0, 0, 7), groups=dim)
            self.conv_dilated_d = nn.Conv3d(dim, dim, kernel_size=(8, 1, 1), stride=(1, 1, 1), padding=(28, 0, 0),
                                            groups=dim, dilation=8)
            self.conv_dilated_h = nn.Conv3d(dim, dim, kernel_size=(1, 8, 1), stride=(1, 1, 1), padding=(0, 28, 0),
                                            groups=dim, dilation=8)
            self.conv_dilated_v = nn.Conv3d(dim, dim, kernel_size=(1, 1, 8), stride=(1, 1, 1), padding=(0, 0, 28),
                                            groups=dim, dilation=8)


        self.conv1 = nn.Conv3d(dim, dim, kernel_size=1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0d(x)
        attn = self.conv0h(attn)
        attn = self.conv0v(attn)
        attn = self.conv_dilated_d(attn)
        attn = self.conv_dilated_h(attn)
        attn = self.conv_dilated_v(attn)

        attn = self.conv1(attn)

        return u * attn

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, patch_size = (4, 4, 4),num_slices=None):
        super().__init__()
        self.dim = dim
        self.mamba = Mamba(
                d_model=dim,
                dim = dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                patch_size = patch_size,
                nslices=num_slices,
        )


    def forward(self, x):
        B, C, H, W, D = x.shape
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_mamba = self.mamba(x)

        x_out = x_mamba + x_skip
        return x_out

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
