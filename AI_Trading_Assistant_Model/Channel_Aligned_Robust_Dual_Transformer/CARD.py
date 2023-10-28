import tensorflow as tf
from .Module.main_module import DualTransformerBlock
from .Module.main_module import Tensor2Seq
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

class Transformer(tf.keras.Model):
    def __init__(
        self,
        seq_length: int,
        output_length: int,
        channel: int,
        patch_size: int,
        strides: int,
        dff: int,
        num_heads: int,
        d_model: int,
        decay: float,
        **kwargs,
    ):
        super().__init__()
        self.TransformerBlock = DualTransformerBlock(seq_length=seq_length, channel=channel, patch_size=patch_size, strides=strides, dff=dff, num_heads=num_heads, d_model=d_model, decay=decay)
        self.T2S = Tensor2Seq(output_length=output_length, channel=channel, d_model=d_model, dff=dff)
        self.channel = channel

    def call(self, x):

        x = self.TransformerBlock(x)
        x = self.T2S(x)

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
    loss = K.min(loss / decay / tf.cast(length, tf.float32))
    return loss