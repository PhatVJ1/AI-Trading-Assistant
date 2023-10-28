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

    self.self_attention1 = MultiHeadAttention.GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)
    
    self.self_attention2 = MultiHeadAttention.GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x1 = x[:,0,:,:]
    x2 = x[:,1,:,:]

    x1 = self.self_attention1(x1)
    x2 = self.self_attention1(x2)

    x1 = x1 * tf.math.sigmoid(x2)
    x2 = x2 * tf.math.sigmoid(x1)

    x1 = tf.expand_dims(x1, axis=1)
    x2 = tf.expand_dims(x2, axis=1)
    x = tf.concat([x1, x2], axis=1)

    x = self.ffn(x)
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
    
    self.cross_attention1 = MultiHeadAttention.CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn_c = FeedForward(d_model, dff)
    self.ffn_o = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x1 = self.cross_attention(x=x[:,0,:,:], context=context[:,0,:,:])
    x2 = self.cross_attention1(x=x[:,1,:,:], context=context[:,1,:,:])

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores
 
    x1 = tf.expand_dims(self.ffn_o(x1), axis=1)
    x2 = tf.expand_dims(self.ffn_c(x2), axis=1)

    x = tf.concat([x1, x2], axis=1)

    return x