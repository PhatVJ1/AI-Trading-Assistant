import tensorflow as tf
from . import Encoder_Decoder as ed
from . import hyperparameter as hp
from tensorflow.keras import backend as K

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
    
    self.conv2d = tf.keras.layers.Conv2D(dff, kernel_size=3, padding="same")

    self.final_layer_c = tf.keras.layers.Dense(target_vocab_size)
    self.final_layer_o = tf.keras.layers.Dense(target_vocab_size)

    self.flatten = tf.keras.layers.Flatten()

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model
    x = self.conv2d(x)

    logits_o = self.flatten(x[:,0,:,:])
    logits_c = self.flatten(x[:,1,:,:])

    # Final linear layer output.
    logits_o = self.final_layer_o(logits_o) # (batch_size, target_len, target_vocab_size)
    logits_c = self.final_layer_c(logits_c)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits_o._keras_mask
      del logits_c._keras_mask
    except AttributeError:
      pass
    
    logits_o = tf.squeeze(logits_o)
    logits_c = tf.squeeze(logits_c)
    output = [logits_o, logits_c]
    
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
    model.compile(optimizer=optimizer.get(), loss=[masked_loss, masked_loss], metrics={"output_1" : custom_metric, "output_2" : custom_metric})
    return model

def masked_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    squared_difference = tf.square(y_true - y_pred)
    masked_loss = mask * squared_difference
    return tf.reduce_mean(masked_loss)

def custom_metric(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), 'float32')
    squared_difference = tf.square(y_true - y_pred)
    masked_squared_difference = mask * squared_difference
    return K.sqrt(K.mean(masked_squared_difference))