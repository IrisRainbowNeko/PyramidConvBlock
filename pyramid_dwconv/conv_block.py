import torch
from torch import nn
from torch.nn import functional as F
from .dwconv_triton import OptimizedDepthwiseConv2d

class SimpleGate(nn.Module):
    def __init__(self, dim: int, kernel_size=13, scale_factor=1., bias=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale = (dim/2)**-0.5
        self.conv_spatial = OptimizedDepthwiseConv2d(dim, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        B, H, W, C = x.shape
        if self.scale_factor != 1.:
            x = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), (int(H*self.scale_factor), int(W*self.scale_factor))).permute(0, 2, 3, 1)
        x = self.conv_spatial(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * (x2/self.scale)
        if self.scale_factor != 1.:
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(H,W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim: int, bias=False, out_bias=True, kernel_size=9, scales=(1.,1/2), inter_ratio=0.5):
        super().__init__()

        inter_dim = int(dim*inter_ratio)
        self.conv_up = nn.Linear(dim, inter_dim, bias=bias)
        self.gates = nn.ModuleList([SimpleGate(inter_dim, kernel_size=kernel_size, scale_factor=scale, bias=bias) for scale in scales])
        self.conv_out = nn.Linear(inter_dim//2*len(scales), dim, bias=out_bias)

        self.conv_pool = nn.Linear(dim, dim, bias=out_bias)

    def forward(self, x):
        x_conv = self.conv_up(x)

        x_conv_list = []
        for gate in self.gates:
            x_conv_list.append(gate(x_conv))

        x_conv = torch.cat(x_conv_list, dim=-1)
        x_conv = self.conv_out(x_conv)

        x_pool = x.mean(dim=(1,2), keepdim=True)
        x_pool = self.conv_pool(x_pool)

        return x_conv + x_pool
    

class SimpleGateT(nn.Module):
    def __init__(self, dim: int, kernel_size=13, scale_factor=1., bias=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale = (dim/2)**-0.5
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size // 2,
                                      bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.scale_factor != 1.:
            x = F.adaptive_avg_pool2d(x, (int(H*self.scale_factor), int(W*self.scale_factor)))
        x = self.conv_spatial(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * (x2/self.scale)
        if self.scale_factor != 1.:
            x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        return x


class ConvBlockT(nn.Module):
    def __init__(self, dim: int, bias=False, out_bias=True, kernel_size=9, scales=(1.,1/2), inter_ratio=0.5):
        super().__init__()

        inter_dim = int(dim*inter_ratio)
        self.conv_up = nn.Conv2d(dim, inter_dim, kernel_size=1, bias=bias)
        self.gates = nn.ModuleList([SimpleGateT(inter_dim, kernel_size=kernel_size, scale_factor=scale, bias=bias) for scale in scales])
        self.conv_out = nn.Conv2d(inter_dim//2*len(scales), dim, kernel_size=1, bias=out_bias)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Conv2d(dim, dim, kernel_size=1, bias=out_bias)

    def forward(self, x):
        x_conv = self.conv_up(x)

        x_conv_list = []
        for gate in self.gates:
            x_conv_list.append(gate(x_conv))

        x_conv = torch.cat(x_conv_list, dim=1)
        x_conv = self.conv_out(x_conv)

        x_pool = self.pool(x)
        x_pool = self.conv_pool(x_pool)

        return x_conv + x_pool
    
def benchmark(batch_size=8, height=224, width=224, channels=64, kernel_size=13, dtype=torch.float16):
    import time
    
    # 创建输入
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()
    
    # 创建我们的实现
    conv_triton = ConvBlock(
        dim=channels,
        kernel_size=kernel_size,
    ).cuda().to(dtype)
    
    # 创建PyTorch的实现（需要转换为BCHW格式）
    x_torch = x.permute(0, 3, 1, 2).clone()  # BHWC -> BCHW
    conv_torch = ConvBlockT(
        dim=channels,
        kernel_size=kernel_size,
    ).cuda().to(dtype)

    #conv_torch.weight.data = conv_triton.weight.permute(2,0,1).unsqueeze(1).data.clone()
    
    # 预热
    for _ in range(10):
        y1 = conv_triton(x)
        #y1.mean().backward()
        y2 = conv_torch(x_torch)
        #y2.mean().backward()
    
    torch.cuda.synchronize()
    
    # 测试我们的实现
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        y = conv_triton(x)
        #y.mean().backward()
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # 测试PyTorch的实现
    start = time.time()
    for _ in range(iterations):
        y = conv_torch(x_torch)
        #y.mean().backward()
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # 结果转换为毫秒
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"批大小={batch_size}, 高={height}, 宽={width}, 通道数={channels}, 卷积核={kernel_size}x{kernel_size}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"加速比: {torch_ms/triton_ms:.2f}x")
    
    return triton_ms, torch_ms

if __name__ == "__main__":
    # 使用案例测试
    batch_size = 4
    height = 64
    width = 64
    channels = 1152
    benchmark(batch_size, height, width, channels, kernel_size=9)
    benchmark(batch_size, height, width, channels, kernel_size=13)
    benchmark(batch_size, height, width, channels, kernel_size=17)