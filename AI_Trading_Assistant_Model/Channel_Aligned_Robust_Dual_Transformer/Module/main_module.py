import tensorflow as tf
from ..layers.Tokenizer import Tokenize
from ..layers.Attention import DualAttentionoverChannels
from ..layers.Attention import DualAttentionoverTokens

#Attention process
class DualTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length: int,
        patch_size: int,
        strides: int,
        dff: int,
        num_layer: int,
        num_heads: int,
        d_model: int,
        decay: float,
        **kwargs,
    ):
        super().__init__()
        self.tokenize = Tokenize(seq_length=seq_length, d_model=d_model, patch_size=patch_size, strides=strides, dff=dff)
        self.DAC = [DualAttentionoverChannels(num_heads=num_heads, key_dim=d_model, num_patch=self.tokenize.pos_emb.num_patch + 1, decay=decay)
                    for i in range(num_layer)]
        self.DAT = [DualAttentionoverTokens(num_heads=num_heads, key_dim=d_model, num_patch=self.tokenize.pos_emb.num_patch + 1, decay=decay)
                    for i in range(num_layer)]
        self.add_norm = [tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization()
        ]) for i in range(num_layer)]
        self.add = tf.keras.layers.Add()

    def call(self, x):
        x = self.tokenize(x=x)
        res = x
        for dac, dat , an in zip(self.DAC, self.DAT, self.add_norm):
          x1 = tf.transpose(dac(tf.transpose(x, perm=(0, 2, 1, 3)), value=tf.transpose(x, perm=(0, 2, 1, 3))), perm=(0, 2, 1, 3))
          x2 = dat(x1, value=x1)
          x = an([x1, x2])
        x = self.add([res, x])

        return x

#Convert attention to target predict sequence
class Tensor2Seq(tf.keras.layers.Layer):
    def __init__(self,*, output_length):
        super().__init__()
        self.MLP = tf.keras.layers.Dense(output_length, activation="linear")

    def call(self, x):
        x_shape = tf.shape(x)
        x = tf.reshape(x, [x_shape[0], x_shape[1], -1])
        x = self.MLP(x)
        return tf.squeeze(x)