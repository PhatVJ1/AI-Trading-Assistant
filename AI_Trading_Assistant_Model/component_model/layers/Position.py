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
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)
    self.open = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=3, padding='same', activation='sigmoid')
    self.close = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=3, padding='same', activation='sigmoid')

  def call(self, x):
    length = x.shape[2]
    open, close = x[:, 0, :], x[:, 1, :]

    open = self.open(open[:, :, tf.newaxis])
    close = self.close(close[:, :, tf.newaxis])

    open = tf.expand_dims(open, axis=1)
    close = tf.expand_dims(close, axis=1)

    x = tf.concat([open, close], axis=1)
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos = tf.expand_dims(self.pos_encoding[tf.newaxis, :length, :], axis=1)
    x = x + tf.concat([pos, pos], axis=1)
    return x
  
class PositionalTracebackEncoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=vocab_size, depth=d_model)
    self.open = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=3, padding='same', activation='sigmoid')
    self.close = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=3, padding='same', activation='sigmoid')
    self.volume = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=3, padding='same', activation='tanh')

  def call(self, x):
    length = x.shape[2]
    open, close, volume = x[:, 0, :], x[:, 1, :], x[:, 2, :]

    open = self.open(open[:, :, tf.newaxis])
    close = self.close(close[:, :, tf.newaxis])
    volume = self.volume(volume[:, :, tf.newaxis])

    open = tf.expand_dims(open + volume, axis=1)
    close = tf.expand_dims(close + volume, axis=1)

    x = tf.concat([open, close], axis=1)
    
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    pos = tf.expand_dims(self.pos_encoding[tf.newaxis, :length, :], axis=1)
    x = x + tf.concat([pos, pos], axis=1)
    return x