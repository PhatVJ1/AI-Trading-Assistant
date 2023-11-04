import tensorflow as tf
from ..layers.Attention import DualAttentionoverTokens
from ..layers.Attention import DualAttentionoverChannels


class AttentionOverTokenDimension(tf.keras.layers.Layer):
    def __init__(
        self,
        channel: int,
        num_heads: int,
        key_dim: int,
        num_patch: int,
        decay: float,
        **kwargs,
    ):
        self.channel = channel
        super().__init__()
        self.list_DAT = [
            DualAttentionoverTokens(num_heads=num_heads, key_dim=key_dim, num_patch=num_patch, decay=decay)
            for _ in range(channel)
        ]

    def call(self, x):
        #Compute attention each token in a channel
        output = tf.expand_dims(self.list_DAT[0](query=x[:, 0, :, :], key=x[:, 0, :, :], value=x[:, 0, :, :]), axis = 1)
        for i in range(1, self.channel):
            output = tf.concat([output, tf.expand_dims(self.list_DAT[i](query=x[:, i, :, :], key=x[:, i, :, :], value=x[:, i, :, :]), axis = 1)], axis=1)
        return output
    
class AttentionOverChannelDimension(tf.keras.layers.Layer):
    def __init__(
        self,
        num_patch: int,
        num_heads: int,
        key_dim: int,
        decay: float,
        **kwargs,
    ):
        super().__init__()
        self.num_patch = num_patch
        self.list_DAC = [
            DualAttentionoverChannels(num_heads=num_heads, key_dim=key_dim, num_patch=num_patch, decay=decay)
            for _ in range(num_patch)
        ]

    def call(self, x):
        #Compute attention each channel per token
        output = tf.expand_dims(self.list_DAC[0](query=x[:, :, 0, :], key=x[:, :, 0, :], value=x[:, :, 0, :]), axis = 2)
        for i in range(1, self.num_patch):
            output = tf.concat([output, tf.expand_dims(self.list_DAC[i](query=x[:, :, i, :], key=x[:, :, i, :], value=x[:, :, i, :]), axis = 2)], axis=2)
        return output