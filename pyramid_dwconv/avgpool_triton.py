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
def _adaptive_avg_pool_kernel(
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
    
    # 空间位置计算（使用更大的块处理中等尺寸空间）
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # (BLOCK_H)
    w_idx = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # (BLOCK_W)
    
    # 动态窗口参数
    h_start = tl.floor((h_idx.to(tl.float16) * in_h) / out_h).to(tl.int32)
    h_end = tl.ceil(((h_idx + 1).to(tl.float16) * (in_h)) / out_h).to(tl.int32)
    h_num = h_end-h_start
    w_start = tl.floor(((w_idx).to(tl.float16) * (in_w)) / out_w).to(tl.int32)
    w_end = tl.ceil(((w_idx + 1).to(tl.float16) * (in_w)) / out_w).to(tl.int32)
    w_num = w_end-w_start

    # 输入特征图起始位置
    batch_offset = pid_batch * in_h * in_w * channels
    
    # 初始化累加器
    sum_acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
    
    # 滑动窗口遍历
    for kh in range(tl.max(h_num)):
        for kw in range(tl.max(w_num)):
            # 计算输入坐标 (保持二维形状)
            current_h = h_start + kh  # (BLOCK_H)
            current_w = w_start + kw  # (BLOCK_W)
            
            # 生成空间掩码 (自动广播为BLOCK_H x BLOCK_W)
            h_mask = (current_h >= 0) & (current_h < h_end)  # (BLOCK_H)
            w_mask = (current_w >= 0) & (current_w < w_end)  # (BLOCK_W)
            valid_mask = h_mask[:, None] & w_mask[None, :]  # (BLOCK_H, BLOCK_W)
            
            # 输入指针计算 (显式广播到三维)
            input_ptrs = (
                batch_offset +
                (current_h[:, None, None] * in_w * channels) +
                (current_w[None, :, None] * channels) + 
                (c_start + tl.arange(0, BLOCK_C))[None, None, :]
            )
            
            # 带掩码的向量化加载
            data = tl.load(
                input_ptr + input_ptrs,
                mask=valid_mask[:, :, None] & c_mask[None, None, :],
                other=0.0
            )
            
            # 累加求和和计数
            sum_acc += data
            count_acc += valid_mask  # 直接使用二维掩码
    
    # 计算平均值
    safe_count = tl.maximum(count_acc, 1).to(tl.float32)[:, :, None]
    avg_result = sum_acc / safe_count
    
    # 输出指针计算
    output_ptrs = (
        pid_batch * out_h * out_w * channels +
        (h_idx[:, None, None] * out_w * channels) +
        (w_idx[None, :, None] * channels) + 
        (c_start + tl.arange(0, BLOCK_C))[None, None, :]
    )
    
    # 生成输出掩码
    output_mask = (
        (h_idx[:, None, None] < out_h) & 
        (w_idx[None, :, None] < out_w) & 
        c_mask[None, None, :]
    )
    
    # 结果存储
    tl.store(output_ptr + output_ptrs, avg_result, mask=output_mask)


class OptimizedAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        # 输入验证
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
        
        # 启动优化后的kernel
        _adaptive_avg_pool_kernel[grid](
            x, output,
            batch, in_h, in_w, channels,
            out_h, out_w,
        )
        return output


# 性能测试
def benchmark(batch_size=8, height=224, width=224, channels=64, scale=0.5, dtype=torch.float16):
    import time
    
    # 创建输入
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()
    
    # 创建我们的实现
    conv_triton = OptimizedAdaptiveAvgPool2d(
        (int(height*scale), int(width*scale))
    ).cuda()
    
    # 创建PyTorch的实现（需要转换为BCHW格式）
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    conv_torch = torch.nn.AdaptiveAvgPool2d(
        (int(height*scale), int(width*scale))
    ).cuda().to(dtype)
    
    # 预热
    for _ in range(10):
        _ = conv_triton(x)
        _ = conv_torch(x_torch)
    
    torch.cuda.synchronize()
    
    # 测试我们的实现
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = conv_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # 测试PyTorch的实现
    start = time.time()
    for _ in range(iterations):
        _ = conv_torch(x_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # 结果转换为毫秒
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"批大小={batch_size}, 高={height}, 宽={width}, 通道数={channels}, scale={scale}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"加速比: {torch_ms/triton_ms:.2f}x")
    
    # 验证结果正确性
    x.requires_grad = True
    out_triton = conv_triton(x)
    out_torch = conv_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"最大绝对误差: {max_diff.item()}")
    
    return triton_ms, torch_ms

if __name__ == "__main__":
    # 使用案例测试
    batch_size = 2
    height = 64
    width = 64
    channels = 1152
    benchmark(batch_size, height, width, channels, scale=1/6)