import tensorflow as tf

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  
  def get_config(self):
    config = super().get_config()
    config.update({
        'mha': self.mha.get_config(),
        'layernorm': self.layernorm.get_config(),
        'add': self.add.get_config(),
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
  
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
class CausalSelfAttention(BaseAttention):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.mha1 = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm1 = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x1 = x[:,0,:,:]
    x2 = x[:,1,:,:]

    x1_attn = self.mha(
        query=x1,
        value=x1,
        key=x1,
        use_causal_mask = True)
    
    x2_attn = self.mha1(
        query=x2,
        value=x2,
        key=x2,
        use_causal_mask = True)
    
    x1 = self.add([x1 * tf.math.sigmoid(x2_attn), x1_attn])
    x2 = self.add([x2 * tf.math.sigmoid(x1_attn), x2_attn])

    x1 = tf.expand_dims(self.layernorm(x1), axis=1)
    x2 = tf.expand_dims(self.layernorm1(x2), axis=1)

    x = tf.concat([x1, x2], axis=1)
    
    return x