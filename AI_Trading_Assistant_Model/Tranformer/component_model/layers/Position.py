import numpy as np
import tensorflow as tf

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalTracebackDecoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)
    self.expanded = tf.keras.layers.Conv1D(vocab_size*d_model, 3, padding="same", activation="sigmoid", groups=vocab_size)

  def mask_compute(self, shape, length):
    mask = np.ones(shape, dtype=bool)
    mask[:, :, int(length):, :] = False
    return mask

  def call(self, x):
    length = x.shape[-1]
    if length < self.vocab_size:
      paddings = tf.constant([[0, 0],[0, 0], [0, self.vocab_size - length]])
      x = tf.pad(x, paddings)
    
    mask = x != 0
    multiples = [1, 1, self.d_model, 1]
    mask = tf.tile(mask[:,:,None,:], multiples)
    mask = tf.transpose(mask, perm=[0, 1, 3, 2])
    x = self.expanded(x)
    x = tf.reshape(x, (-1, x.shape[1], self.vocab_size, self.d_model))
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos = tf.expand_dims(self.pos_encoding, axis=0)
    x = x + pos
    x = tf.where(mask, x, 0)

    return x
  
class PositionalTracebackEncoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)
    self.expanded = tf.keras.layers.Conv1D(vocab_size*d_model, 3, padding="same", activation="sigmoid", groups=vocab_size)

  def call(self, x):
    length = x.shape[-1]
    x = self.expanded(x)
    x = tf.reshape(x, (-1, x.shape[1], self.vocab_size, self.d_model))

    x = tf.math.sigmoid(x[:,0:2,:,:] * x[:,-1:,:,:])
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos = tf.expand_dims(self.pos_encoding[tf.newaxis, :length, :], axis=1)
    paddings = tf.constant([[0, 0], [0, 0], [0, self.vocab_size - length], [0, 0]])
    pos = tf.pad(pos, paddings)
    x = x + pos
    
    return x