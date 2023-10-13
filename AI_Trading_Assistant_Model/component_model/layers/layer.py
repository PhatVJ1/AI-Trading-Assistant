import tensorflow as tf
from . import MultiHeadAttention

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 

    return x
  
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = MultiHeadAttention.GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn_o = FeedForward(d_model, dff)
    self.ffn_c = FeedForward(d_model, dff)

  def call(self, x):
    open = x[:,0,:,:]
    close = x[:,1,:,:]

    open = open + tf.math.sigmoid(close)
    close = close + tf.math.sigmoid(open)

    open = self.self_attention(open)
    close = self.self_attention(close)

    open = tf.expand_dims(self.ffn_o(open), axis=1)
    close = tf.expand_dims(self.ffn_c(close), axis=1)

    x = tf.concat([open, close], axis=1)

    return x
  
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = MultiHeadAttention.CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = MultiHeadAttention.CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn_c = FeedForward(d_model, dff)
    self.ffn_o = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores
 
    open = tf.expand_dims(self.ffn_o(x[:,0,:,:]), axis=1)
    close = tf.expand_dims(self.ffn_c(x[:,1,:,:]), axis=1)

    x = tf.concat([open, close], axis=1)

    return x