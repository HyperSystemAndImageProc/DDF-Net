from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Conv1D, Conv2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten
from tensorflow.keras.layers import Add, Concatenate, Lambda, Input, Permute
from tensorflow.keras.regularizers import L2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply
from attention_models import attention_block
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense, Permute
from tensorflow.keras.regularizers import l2

# Define frequency–space attention module
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, Add


# Define frequency–space attention module
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, Add


# Frequency–space attention module
class FrequencySpaceAttention(Layer):
    def __init__(self, **kwargs):
        super(FrequencySpaceAttention, self).__init__(**kwargs)
        self.conv_A = None
        self.conv_B = None
        self.conv_C = None
        self.alpha = None
        self.dropout = Dropout(0.25)  # Define Dropout layer

    def build(self, input_shape):
        C = input_shape[-1]  # Channel count
        # Define 1x1 conv layers to produce A, B, C; output channels equal input channels
        self.conv_A = Conv2D(filters=C, kernel_size=(1, 1), use_bias=False)
        self.conv_B = Conv2D(filters=C, kernel_size=(1, 1), use_bias=False)
        self.conv_C = Conv2D(filters=C, kernel_size=(1, 1), use_bias=False)
        # Define learnable parameter alpha
        self.alpha = self.add_weight(name='alpha', shape=(1,), initializer='zeros', trainable=True)
        super(FrequencySpaceAttention, self).build(input_shape)

    def call(self, inputs):
        # Get dynamic shape
        B = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        C = tf.shape(inputs)[3]
        w = H * W  # w = H * W

        # Generate A, B, C tensors
        A = self.conv_A(inputs)  # Shape: (batch_size, H, W, C)
        B_mat = self.conv_B(inputs)  # Shape: (batch_size, H, W, C)
        C_mat = self.conv_C(inputs)  # Shape: (batch_size, H, W, C)

        # Reshape to (batch_size, w, C)
        A_reshaped = tf.reshape(A, (B, w, C))  # Shape: [batch_size, w, C]
        B_reshaped = tf.reshape(B_mat, (B, w, C))  # Shape: [batch_size, w, C]
        C_reshaped = tf.reshape(C_mat, (B, w, C))  # Shape: [batch_size, w, C]

        # Compute attention scores
        attn_scores = tf.matmul(B_reshaped, A_reshaped, transpose_b=True)  # Shape: (batch_size, w, w)
        attn_scores = tf.nn.softmax(attn_scores, axis=-1)  # Softmax over the last dimension

        # Compute weighted sum
        E = tf.matmul(attn_scores, C_reshaped)  # Shape: (batch_size, w, C)

        # Reshape back to original (batch_size, H, W, C)
        E = tf.reshape(E, (B, H, W, C))  # Shape: [batch_size, H, W, C]

        # Apply Dropout and residual connection
        E = self.dropout(E)
        output = self.alpha * E + inputs  # Shape: [batch_size, H, W, C]

        return output


# Time-domain attention module
class TimeDomainAttention(Layer):
    def __init__(self, dropout_rate=0.25, **kwargs):
        super(TimeDomainAttention, self).__init__(**kwargs)
        self.beta = self.add_weight(name='beta', shape=(1,), initializer='zeros', trainable=True)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        # tf.print("TimeDomainAttention - inputs shape:", tf.shape(inputs))
        # Assume inputs shape is [batch_size, window_size, 1, F2]
        # Reshape to [batch_size, window_size, F2] by removing the single dimension
        inputs_reshaped = tf.squeeze(inputs, axis=2)  # Shape: [batch_size, window_size, F2]

        # tf.print("TimeDomainAttention - inputs_reshaped shape:", tf.shape(inputs_reshaped))
        # Compute attention weights

        Q = inputs_reshaped
        K = inputs_reshaped
        V = inputs_reshaped

        # Compute attention scores
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)  # Shape: [batch_size, window_size, window_size]

        # Compute attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)  # Shape: [batch_size, window_size, window_size]

        # Compute attention output
        attention_output = tf.matmul(attention_weights, V)  # Shape: [batch_size, window_size, F2]

        # Apply Dropout
        attention_output = self.dropout(attention_output)

        # Restore original shape [batch_size, window_size, 1, F2]
        attention_output = tf.expand_dims(attention_output, axis=2)
        # tf.print("TimeDomainAttention - attention_output shape:", tf.shape(attention_output))
        # Residual connection
        output = self.beta * attention_output + inputs
        # tf.print("TimeDomainAttention - output shape:", tf.shape(output))

        return output



# Combination of frequency–space and time-domain attention modules
def frequency_space_attention(inputs):
    fs_attention = FrequencySpaceAttention()(inputs)
    time_attention = TimeDomainAttention()(fs_attention)
    output = Add()([fs_attention, time_attention])
    return output
def time_domain_attention(inputs):
    # Keep original time-domain attention implementation if used independently
    time_attention = TimeDomainAttention()(inputs)
    return time_attention
def fs_domain_attention(inputs):
    fs_attention = FrequencySpaceAttention()(inputs)
    return  fs_attention
# Build the entire model
def build_model(input_shape, n_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Frequency–space attention module
    fs_attention_output = frequency_space_attention(inputs)

    # Time-domain attention module
    time_attention_output = time_domain_attention(fs_attention_output)

    # Fuse outputs
    fused_output = Add()([fs_attention_output, time_attention_output])

    # Optional classifier
    x = layers.Conv2D(64, (3, 3), activation='relu')(fused_output)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model





def DDFNet_(n_classes, in_chans=22, in_samples=1125, n_windows=5, attention1='mha', attention2='custom',
            eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
            tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
            tcn_activation='elu', fuse='average'):
    input_1 = layers.Input(shape=(1, in_chans, in_samples))  # [None, 1, 22, 1125]
    input_2 = Permute((3, 2, 1))(input_1)  # [None, 1125, 22, 1]



    dense_weightDecay = 0.5
    conv_weightDecay = 0.009
    conv_maxNorm = 0.6
    from_logits = False

    numFilters = eegn_F1
    F2 = numFilters * eegn_D





    # First branch using Conv_block_
    block1 = Conv_block_(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                         kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                         weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                         in_chans=in_chans, dropout=eegn_dropout)
    block1 = Lambda(lambda x: x[:, :, -1, :])(block1)  # [batch_size, features]

    sw_concat = []  # Store outputs of sliding windows
    for i in range(n_windows):
        st = i
        end = tf.shape(block1)[1] - n_windows + i + 1
        block2 = block1[:, st:end, :]  # [batch_size, window_size, F2]


        block2 = attention_block(block2, attention1)  # Assume attention_block is defined
        concat_out = block2  # Or process further as needed
        input_dimension = F2


        block3 = TCN_block_(input_layer=concat_out, input_dimension=input_dimension, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        block3 = Lambda(lambda x: x[:, -1, :])(block3)  # [batch_size, features]

        # Subsequent operations
        if fuse == 'average':
            sw_concat.append(Dense(n_classes, kernel_regularizer=l2(dense_weightDecay))(block3))
        elif fuse == 'concat':
            if i == 0:
                sw_concat = block3
            else:
                sw_concat = Concatenate()([sw_concat, block3])






    # Second branch
    block4 = DSTR_block(input_layer=input_2, F1=eegn_F1, D=eegn_D,
                       kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                       weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                       in_chans=in_chans, dropout=eegn_dropout)

    block4 = Lambda(lambda x: x[:, :, -1, :])(block4)  # [batch_size, features]

    sw_concat_2 = []  # Store sliding-window outputs of second branch

    for i in range(n_windows):
        st = i
        end = tf.shape(block4)[1] - n_windows + i + 1
        block5 = block4[:, st:end, :]  # [batch_size, window_size, F2]

        if attention2 == 'custom':
            # Convert block5 to a 4D tensor
            block5_reshaped = tf.expand_dims(block5, axis=2)  # [batch_size, window_size, 1, F2]
            # Compute frequency–space attention output
            fs_attention_output = frequency_space_attention(block5_reshaped)  # [batch_size, window_size, 1, F2]
            # Compute time-domain attention output
            # time_attention_output1 = time_domain_attention(fs_attention_output)
            time_attention_output = time_domain_attention(fs_attention_output)  # [batch_size, window_size, 1, F2]

            # Fuse outputs of the two modules
            block5 = Add()([fs_attention_output, time_attention_output])  # [batch_size, window_size, 1, F2]
            # Remove extra dimension if needed
            block5 = Lambda(lambda x: tf.squeeze(x, axis=2))(block5)  # [batch_size, window_size, F2]
        else:
            block5 = attention_block(block5, attention2)  # Assume attention_block is defined

        block6 = TCN_block_(input_layer=block5, input_dimension=F2, depth=tcn_depth,
                            kernel_size=tcn_kernelSize, filters=tcn_filters,
                            weightDecay=conv_weightDecay, maxNorm=conv_maxNorm,
                            dropout=tcn_dropout, activation=tcn_activation)
        block6 = Lambda(lambda x: x[:, -1, :])(block6)  # [batch_size, features]

        if fuse == 'average':
            sw_concat_2.append(Dense(n_classes, kernel_regularizer=l2(dense_weightDecay))(block6))
        elif fuse == 'concat':
            if i == 0:
                sw_concat_2 = block6
            else:
                sw_concat_2 = Concatenate()([sw_concat_2, block6])





    # Fuse outputs of the two branches
    if fuse == 'average':
        if len(sw_concat) > 1:
            sw_concat = layers.Average()([*sw_concat, *sw_concat_2])  # [batch_size, n_classes]
        else:
            sw_concat = sw_concat[0]
    elif fuse == 'concat':
        sw_concat = Concatenate()([*sw_concat, *sw_concat_2])
        sw_concat = Dense(n_classes, kernel_regularizer=l2(dense_weightDecay))(sw_concat)

    if from_logits:
        out = Activation('linear', name='linear')(sw_concat)
    else:
        out = Activation('softmax', name='softmax')(sw_concat)

    model = Model(inputs=input_1, outputs=out)
    return model



def DSTR_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay=0.009, maxNorm=0.6, dropout=0.25):


    F2 = F1 * D
    block1 = Conv2D(F1, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(input_layer)
    block1 = Conv2D(F1, (kernLength, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block1)
    block1 = Conv2D(F1, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block1)
    block1 = BatchNormalization(axis=-1)(block1)  # bn_axis = -1 if data_format() == 'channels_last' else 1
    block1 = Add()([block1, input_layer])




    block2 = Conv2D(F1, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block1)
    block2 = DepthwiseConv2D((1, in_chans),
                             depth_multiplier=D,
                             data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                             use_bias=False)(block2)
    block2 = Conv2D(F2, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block2)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8, 1), data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)



    block3 = Conv2D(F2, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block2)
    block3 = Conv2D(F2, (16, 1),
                    data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False, padding='same')(block3)
    block3 = Conv2D(F2, (1, 1), padding='same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint=max_norm(maxNorm, axis=[0, 1, 2]),
                    use_bias=False)(block3)
    block3 = BatchNormalization(axis=-1)(block3)
    block3 = Add()([block3, block2])
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize, 1), data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3




def Conv_block_(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22,
                weightDecay = 0.009, maxNorm = 0.6, dropout=0.25):
    """ Conv block (Conv_block)

        Notes
        -----
        Uses different regularization methods.
    """

    F2= F1*D
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same', data_format='channels_last',
                    kernel_regularizer=L2(weightDecay),

                    # In a Conv2D layer with data_format="channels_last", the weight tensor has shape
                    # (rows, cols, input_depth, output_depth); set axis to [0, 1, 2]
                    # to constrain weights of each filter (rows, cols, input_depth).
                    kernel_constraint = max_norm(maxNorm, axis=[0,1,2]),
                    use_bias = False)(input_layer)
    block1 = BatchNormalization(axis = -1)(block1)  # bn_axis = -1 if data_format() == 'channels_last' else 1

    block2 = DepthwiseConv2D((1, in_chans),
                             depth_multiplier = D,
                             data_format='channels_last',
                             depthwise_regularizer=L2(weightDecay),
                             depthwise_constraint  = max_norm(maxNorm, axis=[0,1,2]),
                             use_bias = False)(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)

    block3 = Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            kernel_regularizer=L2(weightDecay),
                            kernel_constraint = max_norm(maxNorm, axis=[0,1,2]),
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)

    block3 = AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)
    return block3


def TCN_block_(input_layer,input_dimension,depth,kernel_size,filters, dropout,
               weightDecay = 0.009, maxNorm = 0.6, activation='relu'):
    """ TCN_block from Bai et al. (2018)
        Temporal Convolutional Network (TCN)

        Notes
        -----
        Uses different regularization methods.
    """

    block = Conv1D(filters, kernel_size=kernel_size, dilation_rate=1, activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),

                    padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),

                    padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),

                    padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)

    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),

                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block) 
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                    kernel_regularizer=L2(weightDecay),
                    kernel_constraint = max_norm(maxNorm, axis=[0,1]),

                    padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out