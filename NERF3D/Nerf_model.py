
# Second part
# Creating the model of MLP each layer, connection
"""
Input---->rays and directions(after encoding) , output ----->RGB and Volume density(sigma)
"""

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate

"""
define model production
xyz and dir are the dimensions of positional encoding for xyz and direction vector
During the process of create model, use the camera coordinate system as a reference
input ray --> layer1 --> layer2 --> layer3 --> layer4 --> layer5(concatenation layer4+input ray)....
coarse model and fine model are two  MLP models
"""


def MLP_model(xyz, dir, dense_unit, batch_size):
    # input 1
    input_ray = Input(shape=(None, None, None, 6 * xyz + 3), batch_size=batch_size)

    # input2 build input layer for direction of the rays
    input_dir = Input(shape=(None, None, None, 6 * dir + 3), batch_size=batch_size)
    # -------------------------------------------------------------
    # produce input for the MLP
    # Pass the ray through model(7 dense layer firstly)
    x = input_ray
    # layer 1 ----> layer 7, concatenation between dense layer 4 and dense layer 5
    for a in range(8):
        # build a dense full-connected layer
        x = Dense(units=dense_unit, activation="relu")(x)

        # According to the whole model's structure, between layer 4 and layer 5 exist a skip layer
        if a % 4 == 0 and a > 0:
            # Add concatenation with input ray, layer5 is a concatenation layer
            x = concatenate([x, input_ray], axis=-1)
    # -------------------------------------------------------------
    # Obtain the Volume density value (sigma), it does not depend on direction, dense layer 8
    volume_density = Dense(units=1, activation="relu")(x)
    # --------------------------------------------------------------
    # obtain the  new_output, this new output is output of Dense 9
    new_output = Dense(units=dense_unit)(x)
    # --------------------------------------------------------------
    # concatenate the new_output with the direction input , then pass it through two dense layer
    new_output_1 = concatenate([new_output, input_dir], axis=-1)
    # layer 10
    x = Dense(units=dense_unit, activation="relu")(new_output_1)
    # layer 11
    x = Dense(units=dense_unit//2, activation="relu")(x)
    # --------------------------------------------------------------
    # Obtain the rgb value, sigmoid control RGB in 0 to 1
    RGB = Dense(units=3, activation="sigmoid")(x)
    # ---------------------------------------------------------------
    # Eventually model created
    nerf_model = Model(inputs=[input_ray, input_dir], outputs=[RGB, volume_density])
    return nerf_model



