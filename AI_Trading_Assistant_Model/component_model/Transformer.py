import tensorflow as tf
from . import Encoder_Decoder as ed
from . import hyperparameter as hp

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = ed.Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = ed.Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer_c = tf.keras.layers.Dense(target_vocab_size)
    self.final_layer_o = tf.keras.layers.Dense(target_vocab_size)

    self.flatten = tf.keras.layers.Flatten()

    self.global_dense = tf.keras.layers.Dense(dff, activation='linear')

    self.dense = [
      tf.keras.layers.Dense(1, activation='linear')
      for _ in range(target_vocab_size * 2)
    ]

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits_o = tf.expand_dims(self.final_layer_o(x[:,0,:,:]), axis=1)  # (batch_size, target_len, target_vocab_size)
    logits_c = tf.expand_dims(self.final_layer_c(x[:,1,:,:]), axis=1)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits_o._keras_mask
      del logits_c._keras_mask
    except AttributeError:
      pass
    
    logits_o = self.flatten(logits_o)
    logits_c = self.flatten(logits_c)

    logists = tf.concat([logits_o, logits_c], axis=1)
    logists = self.global_dense(logists)

    open = tf.expand_dims(self.dense[0](logists), axis=-1)
    close = tf.expand_dims(self.dense[1](logists), axis=-1)
    
    output = tf.concat([open, close], axis=1)

    for i in range(1, int(len(self.dense) / 2)):
      open = tf.expand_dims(self.dense[i*2](logists), axis=-1)
      close = tf.expand_dims(self.dense[i*2 + 1](logists), axis=-1)
      output = tf.concat([output, tf.concat([open, close], axis=1)], axis=-1)

    # Return the final output and the attention weights.
    return output


class model():
  def __init__(self, *, input_shape, num_layers, d_model, num_heads, dff,
               dropout_rate=0.1, warmup_steps = 4000, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9):
    self.input_shape = [(1,) + input_shape[0], (1,) + input_shape[1]]
    self.num_layers = num_layers
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.input_vocab_size = self.input_shape[0][-1]
    self.target_vocab_size = self.input_shape[1][-1]
    self.dropout_rate = dropout_rate
    self.warmup_steps = warmup_steps
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

  def build_model(self):
    model = Transformer(num_layers=self.num_layers, d_model=self.d_model, num_heads=self.num_heads, dff=self.dff,
                        input_vocab_size=self.input_vocab_size, target_vocab_size=self.target_vocab_size, dropout_rate=self.dropout_rate)
    model.build(self.input_shape)
    optimizer = hp.optimizer(self.d_model, self.warmup_steps, self.beta_1, self.beta_2, self.epsilon)
    model.compile(optimizer=optimizer.get(), loss='mse', metrics=['mse', 'mae'])

    return model