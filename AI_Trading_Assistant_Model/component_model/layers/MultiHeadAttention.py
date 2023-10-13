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
    self.layernorm1 = tf.keras.layers.LayerNormalization()

  def call(self, x):
    open = x[:,0,:,:]
    close = x[:,1,:,:]

    open_attn = self.mha(
        query=open,
        value=open,
        key=open,
        use_causal_mask = True)
    
    close_attn = self.mha(
        query=close,
        value=close,
        key=close,
        use_causal_mask = True)
    
    open = self.add([open, open_attn])
    close = self.add([close, close_attn])

    open = tf.expand_dims(self.layernorm(open), axis=1)

    close = tf.expand_dims(self.layernorm1(close), axis=1)

    x = tf.concat([open, close], axis=1)
    
    return x