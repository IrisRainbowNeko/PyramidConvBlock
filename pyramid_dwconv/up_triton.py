import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 32, 'BLOCK_W': 32, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 128}, num_warps=8),
    ],
    key=['batch', 'in_h', 'in_w', 'channels', 'out_h', 'out_w'],
)
@triton.jit
def _bilinear_interp_kernel(
    input_ptr, output_ptr,
    batch, in_h, in_w, channels,
    out_h, out_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 三维并行化
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_bc = tl.program_id(2)
    
    # 分解组合维度
    num_channel_blocks = tl.cdiv(channels, BLOCK_C)
    pid_batch = pid_bc // num_channel_blocks
    pid_c = pid_bc % num_channel_blocks
    
    if pid_batch >= batch or pid_c * BLOCK_C >= channels:
        return
    
    # 通道掩码
    c_start = pid_c * BLOCK_C
    c_mask = (tl.arange(0, BLOCK_C) < channels - c_start)
    
    # 输出空间位置
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # 计算输入浮点坐标
    y_scale = in_h.to(tl.float32) / out_h
    x_scale = in_w.to(tl.float32) / out_w
    
    y = (h_idx.to(tl.float32) + 0.5) * y_scale - 0.5
    x = (w_idx.to(tl.float32) + 0.5) * x_scale - 0.5
    y = tl.maximum(y,0)
    x = tl.maximum(x,0)
    
    # 计算四个邻近点坐标
    y_low = tl.floor(y).to(tl.int32)
    y_low = tl.minimum(tl.maximum(y_low, 0), in_h - 1)
    y_high = tl.minimum(y_low + 1, in_h - 1)
    dy = y - y_low.to(tl.float32)
    
    x_low = tl.floor(x).to(tl.int32)
    x_low = tl.minimum(tl.maximum(x_low, 0), in_w - 1)
    x_high = tl.minimum(x_low + 1, in_w - 1)
    dx = x - x_low.to(tl.float32)
    
    # 计算权重 (广播到三维)
    w_lt = (1 - dx[None, :, None]) * (1 - dy[:, None, None])  # (BLOCK_H, BLOCK_W, 1)
    w_rt = dx[None, :, None] * (1 - dy[:, None, None])
    w_lb = (1 - dx[None, :, None]) * dy[:, None, None]
    w_rb = dx[None, :, None] * dy[:, None, None]
    
    # 输入指针计算
    batch_offset = pid_batch * in_h * in_w * channels
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    
    # 四个邻近点的指针
    ptrs_lt = (
        batch_offset +
        (y_low[:, None, None] * in_w * channels) +
        (x_low[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_rt = (
        batch_offset +
        (y_low[:, None, None] * in_w * channels) +
        (x_high[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_lb = (
        batch_offset +
        (y_high[:, None, None] * in_w * channels) +
        (x_low[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    ptrs_rb = (
        batch_offset +
        (y_high[:, None, None] * in_w * channels) +
        (x_high[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    
    # 带掩码加载数据
    data_lt = tl.load(input_ptr + ptrs_lt, mask=c_mask[None, None, :], other=0.0)
    data_rt = tl.load(input_ptr + ptrs_rt, mask=c_mask[None, None, :], other=0.0)
    data_lb = tl.load(input_ptr + ptrs_lb, mask=c_mask[None, None, :], other=0.0)
    data_rb = tl.load(input_ptr + ptrs_rb, mask=c_mask[None, None, :], other=0.0)
    
    # 插值计算
    interp = (
        data_lt * w_lt +
        data_rt * w_rt +
        data_lb * w_lb +
        data_rb * w_rb
    )
    
    # 输出指针和掩码
    output_ptrs = (
        pid_batch * out_h * out_w * channels +
        (h_idx[:, None, None] * out_w * channels) +
        (w_idx[None, :, None] * channels) +
        c_offsets[None, None, :]
    )
    output_mask = (
        (h_idx[:, None, None] < out_h) & 
        (w_idx[None, :, None] < out_w) & 
        c_mask[None, None, :]
    )
    
    # 结果存储
    tl.store(output_ptr + output_ptrs, interp.to(output_ptr.dtype.element_ty), mask=output_mask)

class OptimizedBilinearInterp2d(torch.nn.Module):
    def __init__(self, output_size, align_corners=False):
        super().__init__()
        self.output_size = output_size
        self.align_corners = align_corners
        
    def forward(self, x):
        # 输入验证 (BHWC格式)
        assert x.dim() == 4, "输入必须是BHWC格式的四维张量"
        batch, in_h, in_w, channels = x.shape
        
        # 解析输出尺寸
        if isinstance(self.output_size, int):
            out_h = out_w = self.output_size
        else:
            out_h, out_w = self.output_size
            
        # 输出张量初始化
        output = torch.empty((batch, out_h, out_w, channels), 
                           device=x.device, dtype=x.dtype)
        
        # 动态网格划分
        def grid(meta):
            return (
                triton.cdiv(out_h, meta['BLOCK_H']),
                triton.cdiv(out_w, meta['BLOCK_W']),
                batch * triton.cdiv(channels, meta['BLOCK_C'])
            )
        
        # 启动kernel
        _bilinear_interp_kernel[grid](
            x, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
        )
        return output

# 正确性验证和性能测试函数
def benchmark_interp(batch=2, in_h=64, in_w=64, channels=1152, scale=3.0, dtype=torch.float16):
    import time
    
    # 创建输入
    x = torch.randn(batch, in_h, in_w, channels, dtype=dtype).cuda()
    
    # 创建我们的实现
    interp_triton = OptimizedBilinearInterp2d(
        (int(in_h*scale), int(in_w*scale))
    ).cuda()
    
    # 创建PyTorch实现 (需要转换为BCHW格式)
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    interp_torch = torch.nn.Upsample(
        scale_factor=scale, 
        mode='bilinear',
        align_corners=False
    ).cuda().to(dtype)
    
    # 预热
    for _ in range(10):
        _ = interp_triton(x)
        _ = interp_torch(x_torch)
    
    torch.cuda.synchronize()
    
    # 测试性能
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = interp_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        _ = interp_torch(x_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # 结果转换为毫秒
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"批大小={batch}, 输入尺寸={in_h}x{in_w}, 通道数={channels}, 缩放因子={scale}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"加速比: {torch_ms/triton_ms:.2f}x")
    
    # 验证结果正确性
    out_triton = interp_triton(x)
    out_torch = interp_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC

    # print(out_triton.view(12,12))
    # print(out_torch.view(12,12))

    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"最大绝对误差: {max_diff.item()}")
    
    return triton_ms, torch_ms

if __name__ == "__main__":
    benchmark_interp(batch=2, in_h=64, in_w=64, channels=1152, scale=3.0)