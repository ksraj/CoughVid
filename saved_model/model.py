import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import np_utils



max_length = 1292

def build_model(img_width = int(max_length/4),img_height = 41):
    # Inputs to the model

    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    
    # First conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2),strides = 2, name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), strides = 2, name="pool2")(x)

    # Third conv block
    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)

    # Fourth conv block
    x = layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv4",
    )(x)

    x = layers.MaxPooling2D((1, 2), name="pool4")(x)

    # Fifth conv block
    x = layers.Conv2D(
        512,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv5",
    )(x)

    x = layers.BatchNormalization(momentum = 0.8, name="BatchNormalization_1")(x)

    # Sixth conv block
    x = layers.Conv2D(
        512,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv6",
    )(x)

    x = layers.BatchNormalization(momentum = 0.8, name="BatchNormalization_2")(x)

    x = layers.MaxPooling2D((1, 2), name="pool6")(x)

    # Seventh conv block
    x = layers.Conv2D(
        512,
        (2, 2),
        activation="relu",
        kernel_initializer="he_normal",
        padding="valid",
        name="Conv7",
    )(x)

    # The number of filters in the last layer is 512. Reshape accordingly before
    # passing the output to the RNN part of the model

    new_shape = (int(max_length/16)-1,512) # downsampling 
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    def attention_rnn(inputs):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        timestep = int(inputs.shape[1])
        a = layers.Permute((2, 1))(inputs) #Permutes the dimensions of the input according to a given pattern.
        a = layers.Dense(timestep, activation='softmax')(a) #// Alignment Model + Softmax
        a = layers.Lambda(lambda x: keras.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
        a_probs = layers.Permute((2, 1), name='attention_vec')(a)
        output_attention_mul = layers.multiply([inputs, a_probs], name='attention_mul') #// Weighted Average 
        return output_attention_mul

    x = attention_rnn(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x  =layers.Flatten()(x)
    # Output layer
    
    x = layers.BatchNormalization(momentum = 0.8)(x)
    x = layers.Dense(512 , activation="relu")(x)
    x = layers.Dense(256 , activation="relu")(x) # 
    
    y_pred = layers.Dense(2 , activation="softmax", name="last_dense")(x) # y pred
    model = keras.models.Model(inputs=input_img, outputs=y_pred, name="model")
    
    return model