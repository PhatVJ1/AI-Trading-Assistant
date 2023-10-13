import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
class optimizer():
  def __init__(self, d_model, warmup_steps = 4000, beta_1=0.9, beta_2=0.98, epsilon=1e-9):
    self.optimizer = optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model, warmup_steps), beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
    
  def get(self):
    return self.optimizer


