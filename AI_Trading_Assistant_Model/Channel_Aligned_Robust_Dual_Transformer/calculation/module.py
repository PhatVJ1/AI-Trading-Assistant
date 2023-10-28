import tensorflow as tf

class EMA_Layer(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, **kwargs):
        super(EMA_Layer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, x):
        split = tf.split(x, x.shape[-2], axis=-2)
        ema_tensor = split[0]
        for i in range(1, x.shape[-2]):
            ema_tensor = tf.concat([ema_tensor, split[i] * self.alpha + ema_tensor[... , -1:, :] * (1 - self.alpha)], axis=-2)

        return ema_tensor

class CustomSoftmax(tf.keras.layers.Layer):
    def __init__(self, axis=[-2, -1], **kwargs):
        super(CustomSoftmax, self).__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        e = tf.exp(x)
        sum = tf.expand_dims(tf.expand_dims(tf.reduce_sum(e, axis=self.axis), axis=-1), axis=-1)
        return e / sum
