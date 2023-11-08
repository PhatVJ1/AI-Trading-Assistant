import tensorflow as tf
from keras.src.utils import tf_utils
from keras.src.layers import core

from ..calculation.module import EMA_Layer
from ..calculation.module import CustomSoftmax

class DualAttentionoverTokens(tf.keras.layers.MultiHeadAttention):
    def __init__(self, num_heads: int, key_dim: int, num_patch: int, decay: int, **kwargs):
        super().__init__(num_heads, key_dim, **kwargs)
        self.num_patch = num_patch
        self._MLP_query = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="gelu"),
            tf.keras.layers.Dense(key_dim, activation="linear")
        ])
        self._MLP_key = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="gelu"),
            tf.keras.layers.Dense(key_dim, activation="linear")
        ])
        self._MLP_value = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="gelu"),
            tf.keras.layers.Dense(key_dim, activation="linear")
        ])
        self.EMA = EMA_Layer(alpha=decay)
        self.softmax = CustomSoftmax()
        self._MLP_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(key_dim, activation="gelu"),
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(key_dim, activation="linear")
        ])
        self._MLP_projection1 = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(key_dim, activation="gelu"),
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(key_dim, activation="linear")
        ])
        self.add_norm = tf.keras.Sequential([
            tf.keras.layers.Add(),
            tf.keras.layers.LayerNormalization()
        ])

    def _build_from_signature(self, query, value, key=None):
        self._built_from_signature = True
        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)

        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
            free_dims = self._query_shape.rank - 1
            self._query_dense = core.EinsumDense(
                "bctd,chtd->bchtd",
                output_shape= [self._query_shape[1], self._num_heads, self.num_patch, self._key_dim],
                bias_axes="htd",
                name="query",
                **self._get_common_kwargs_for_sublayer(),
            )
            self._key_dense = core.EinsumDense(
                "bctd,chtd->bchtd",
                output_shape= [self._query_shape[1], self._num_heads, self.num_patch, self._key_dim],
                bias_axes="htd",
                name="key",
                **self._get_common_kwargs_for_sublayer(),
            )
            self._value_dense = core.EinsumDense(
                "bctd,chtd->bchtd",
                output_shape= [self._query_shape[1], self._num_heads, self.num_patch, self._key_dim],
                bias_axes="htd",
                name="value",
                **self._get_common_kwargs_for_sublayer(),
            )

            # Builds the attention computations for multi-head dot product
            # attention.  These computations could be wrapped into the keras
            # attention layer once it supports mult-head einsum computations.
            self._build_attention(6)
            self._output_dense = self._make_output_dense(
                self._get_common_kwargs_for_sublayer(),
                "attention_output",
            )
    #(5, 8, 3, 49, 128)
    def _make_output_dense(self, common_kwargs, name=None):
        return core.EinsumDense(
            "bchtd,hctd->bctd",
            output_shape=[None] + self._query_shape[2:],
            bias_axes='d',
            name=name,
            **common_kwargs,
        )

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        attention_scores1 = tf.einsum("...NK,...NO->...KO", query, key) / tf.sqrt(tf.cast(self.num_patch, tf.float32))
        attention_scores1 = self.softmax(attention_scores1)

        query = self.EMA(query)
        key = self.EMA(key)

        attention_scores = tf.einsum("...HCKO,...HCNO->...HCKN", query, key) / tf.sqrt(tf.cast(self._key_dim, tf.float32))
        attention_scores = self.softmax(attention_scores)

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_scores1_dropout = self._dropout_layer(
            attention_scores1
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            "...HCKN,...HCNO->...HCKO", attention_scores_dropout, value
        )
        attention_output1 = tf.einsum(
            "...HCDK,...HCNK->...HCND", attention_scores1_dropout, value
        )

        attention_output = self._MLP_projection(attention_output)
        attention_output1 = self._MLP_projection1(attention_output1)

        output = self.add_norm([attention_output, attention_output1])
        return output, attention_output, attention_output1

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        if not self._built_from_signature:
            self.num_patch = query.shape[2]
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        query = self._MLP_query(query)

        key = self._MLP_key(key)

        value = self._MLP_value(value)

        query = self._query_dense(query)

        key = self._key_dense(key)

        value = self._value_dense(value)

        attention_output, attention_scores, attention_scores1 = self._compute_attention(
            query, key, value, attention_mask, training
        )

        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores, attention_scores1
        return attention_output
    
class DualAttentionoverChannels(DualAttentionoverTokens):
    def __init__(self, num_heads: int, key_dim: int, num_patch: int, decay: int, **kwargs):
        super().__init__(num_heads, key_dim, num_patch, decay, **kwargs)
        self.DP_K = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(tf.math.ceil(key_dim / 10), activation="linear")
        ])
        self.DP_V = tf.keras.Sequential([
            tf.keras.layers.Dense(num_patch * num_heads, activation="linear"),
            tf.keras.layers.Dense(tf.math.ceil(key_dim / 10), activation="linear")
        ])

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        attention_scores1 = tf.einsum("...NK,...NO->...KO", query, key) / tf.sqrt(tf.cast(self.num_patch, tf.float32))
        attention_scores1 = self.softmax(attention_scores1)

        query = self.EMA(query)

        PK = self.DP_K(key)
        PV = self.DP_V(value)

        PK = self.softmax(PK)
        PV = self.softmax(PV)

        key = tf.einsum("...CNR,...CND->...RND", PK, key)
        key = self.EMA(key)

        value = tf.einsum("...CNR,...CND->...RND", PV, value)

        attention_scores = tf.einsum("...CKO,...RNO->...CKN", query, key) / tf.sqrt(tf.cast(self._key_dim, tf.float32))
        attention_scores = self.softmax(attention_scores)

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_scores1_dropout = self._dropout_layer(
            attention_scores1
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            "...CKN,...RND->...CKD", attention_scores_dropout, value
        )

        attention_output1 = tf.einsum(
            "...CDK,...RNK->...CND", attention_scores1_dropout, value
        )

        attention_output = self._MLP_projection(attention_output)
        attention_output1 = self._MLP_projection1(attention_output1)

        output = self.add_norm([attention_output, attention_output1])
        return output, attention_output, attention_output1