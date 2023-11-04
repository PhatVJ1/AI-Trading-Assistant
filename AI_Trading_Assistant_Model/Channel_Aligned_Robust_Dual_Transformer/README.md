Author: Nguyễn Thanh Quang

References:
- Wang Xue∗, Tian Zhou∗, Qingsong Wen, Jinyang Gao, Bolin Ding, Rong Jin. Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer, 2023

Package requirements
- Tensorflow

Model create and build:

```python
from Channel_Aligned_Robust_Dual_Transformer import CARD

model = CARD.Transformer(seq_length=50, output_length=3, channel=2, patch_size=3, strides=1, dff=10, num_heads=8, d_model=128, decay=0.8)
CARD.model_builder(model, input[:1])

#Then you can use compile model with signal_decay_loss in CARD and traing model
#Label of model must be a list contains matrixs of each channel
```

