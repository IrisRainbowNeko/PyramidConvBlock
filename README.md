# Fast Depth-Wise Conv2D with Triton

## Install
```bash
pip install -r requirements.txt
```

## Usage
```python
from pyramid_dwconv import ConvBlock

x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()
conv_block = ConvBlock( # weight: [k1,k2,C]
    dim=channels,
    kernel_size=kernel_size,
).cuda().to(dtype)

y = conv_block(x) # [B,H,W,C]
```