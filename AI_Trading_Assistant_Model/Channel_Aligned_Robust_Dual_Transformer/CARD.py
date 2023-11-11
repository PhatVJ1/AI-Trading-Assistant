import tensorflow as tf
from .Module.main_module import DualTransformerBlock
from .Module.main_module import Tensor2Seq
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from tensorflow.keras import backend as K

class Transformer(tf.keras.Model):
    def __init__(
        self,
        seq_length: int,
        output_length: int,
        patch_size: int,
        strides: int,
        dff: int,
        num_heads: int,
        d_model: int,
        decay: float,
        num_layer: int,
        **kwargs,
    ):
        super().__init__()
        self.TransformerBlock = DualTransformerBlock(seq_length=seq_length, patch_size=patch_size, strides=strides, dff=dff, num_layer=num_layer, num_heads=num_heads, d_model=d_model, decay=decay)
        self.T2S = Tensor2Seq(output_length=output_length, dff=dff)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x_mean = K.mean(x, axis=-2, keepdims=True)
        x_std = K.std(x, axis=-2, keepdims=True)
        x = (x - x_mean)/(x_std + 1e-4)

        x = self.TransformerBlock(x)
        x = self.T2S(x)

        x = x * (tf.transpose(x_std, perm=(0, 2, 1)) + 1e-4) + tf.transpose(x_mean, perm=(0, 2, 1))

        return x
    
def model_builder(model, input):
    model(input)

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_epoch = None
        self.previous_val_loss = None
        self.min = 1e9

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        current_val_loss = logs.get('val_loss')
        if self.previous_val_loss is not None:
            print(f'\nVal_loss difference compared to previous model: {self.previous_val_loss - current_val_loss}')
        self.previous_val_loss = current_val_loss
        self.save_model(self.filepath, overwrite=True)

    def save_model(self, filepath, overwrite=True, options=None):
        if 'val_loss' in self.model.history.history and self.previous_val_loss is not None and self.min > self.previous_val_loss:
            print(f"{self.min} compare to {self.previous_val_loss}")
            self.min = self.previous_val_loss
            if self.current_epoch % 2 == 0:
                filepath += '_odd.h5'
            else:
                filepath += '_even.h5'
            self.model.save_weights(filepath, overwrite=overwrite, options=options)
            print(f'\nModel saved weights at {filepath}')

def signal_decay_loss(y_true, y_pred):
    length = len(y_true)
    seq_length = y_true.shape[-1]
    loss = K.sum(K.abs(y_true - y_pred), axis=0)
    decay = tf.range(1, seq_length + 1, dtype=tf.float32)
    decay = K.pow(decay, -0.5)
    loss = K.min(loss / tf.expand_dims(decay, axis = 0) / tf.cast(length, tf.float32), axis=1)
    return K.mean(loss)