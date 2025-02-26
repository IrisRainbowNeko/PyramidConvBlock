# Fast Depth-Wise Conv2D with Triton

## Install
```bash
pip install -r requirements.txt
```

## Speed

### Forward Only
| Kernel | Torch   | Ours    | SpeedUp |
|--------|---------|---------|---------|
| 9x9    | 3.533ms | 0.728ms | 4.85x   |
| 13x13  | 6.190ms | 0.911ms | 6.79x   |
| 17x17  | 9.667ms | 1.172ms | 8.25x   |

### Full
| Kernel | Torch    | Ours    | SpeedUp |
|--------|----------|---------|---------|
| 9x9    | 9.293ms  | 4.096ms | 2.27x   |
| 13x13  | 19.293ms | 5.519ms | 3.50x   |
| 17x17  | 37.713ms | 8.429ms | 4.47x   |

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