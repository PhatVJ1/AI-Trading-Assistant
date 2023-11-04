import tensorflow as tf

#Embedding input for the Transformer model family
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self,*, seq_length, d_model, patch_size, strides, dff):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.patch_size = patch_size
        self.strides = strides
        self.num_patch = (seq_length - patch_size) // strides + 1
        self.MLP_P2D = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model, activation="linear")
        ])
        self.MLP_Pos = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation="tanh"),
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model, activation="linear")
        ])

    def build(self, input_shape):
        #Create the initial position matrix
        self.position = self.add_weight(name="position", shape=(input_shape[-1], self.num_patch),
                                 initializer=tf.initializers.Constant(tf.stack([tf.range(1., self.num_patch + 1)] * input_shape[-1])),
                                 trainable=False)

    def call(self, x):
        #Create a tensor contains all patchs per window with stride
        patches = tf.expand_dims(x[:, 0:self.patch_size, :], axis=0)
        for i in range(1, self.num_patch):
            patches = tf.concat([patches, tf.expand_dims(x[:, i * self.strides : i * self.strides + self.patch_size ,:], axis=0)], axis=0)

        #(tokens, batch, patch_size, channel) -> (batch, channel, tokens, patch_size)
        x = tf.transpose(patches, perm = (1, 3, 0, 2))
        #(batch, channel, tokens, patch_size) -> (batch, channel, tokens, d_model)
        x = self.MLP_P2D(x)
        #Compute position embedding with size (batch, channel, token, d_model)
        position_embedding = self.MLP_Pos(self.position[:,:,tf.newaxis])
        #Add position
        x = tf.add(x, position_embedding)
        return x
    
#Compute relationship between tokens per channel
class ClsCompute(tf.keras.layers.Layer):
    def __init__(self,*, d_model):
        super().__init__()
        self.cls_compute = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1, padding="same", strides=1, activation="sigmoid")

    def call(self, x):
        x = tf.reshape(x, [-1 , x.shape[-3], x.shape[-2] * x.shape[-1]])
        x = self.cls_compute(x)
        return x
    
#Tokenize input for CARD
class Tokenize(tf.keras.layers.Layer):
    def __init__(self,*, seq_length, d_model, patch_size, strides, dff):
        super().__init__()
        self.pos_emb = PositionEmbedding(seq_length=seq_length, d_model=d_model, patch_size=patch_size, strides=strides, dff=dff)
        self.cls_tokenize = ClsCompute(d_model=d_model)

    def call(self, x):
        x = self.pos_emb(x)
        cls = tf.expand_dims(self.cls_tokenize(x), axis=2)
        return tf.concat([cls, x], axis=2)