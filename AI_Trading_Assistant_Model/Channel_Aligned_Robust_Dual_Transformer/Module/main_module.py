import tensorflow as tf
from ..layers.Tokenizer import Tokenize
from .AttentionDimension import AttentionOverChannelDimension
from .AttentionDimension import AttentionOverTokenDimension

#Attention process
class DualTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length: int,
        channel: int,
        patch_size: int,
        strides: int,
        dff: int,
        num_heads: int,
        d_model: int,
        decay: float,
        **kwargs,
    ):
        super().__init__()
        self.tokenize = Tokenize(seq_length=seq_length, d_model=d_model, patch_size=patch_size, strides=strides, dff=dff)
        self.attentionchannel = AttentionOverChannelDimension(num_patch=self.tokenize.pos_emb.num_patch + 1, num_heads=num_heads, key_dim=d_model, decay=decay)
        self.attentiontoken = AttentionOverTokenDimension(channel=channel, num_heads=num_heads, key_dim=d_model, num_patch=self.tokenize.pos_emb.num_patch + 1, decay=decay)
        self.add_norm = tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization()
        ])
        self.add = tf.keras.layers.Add()

    def call(self, x):
        x = self.tokenize(x=x)
        x_c = self.attentionchannel(x)
        x_t = self.attentiontoken(x_c)

        #Combine attention per channels and attention per tokens
        an = self.add_norm([x_c, x_t])
        #Add residual
        x = self.add([x, an])

        return x

#Convert attention to target predict sequence
class Tensor2Seq(tf.keras.layers.Layer):
    def __init__(self,*, output_length, channel, d_model, dff):
        super().__init__()
        self.channel = channel
        self.MLP = [tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model, activation="gelu"),
            tf.keras.layers.Dense(output_length, activation="linear")
        ]) for _ in range(channel)]

    def call(self, x):
        output = [
            self.MLP[i](x[:,i,:,:])
            for i in range(self.channel)
        ]
        return output