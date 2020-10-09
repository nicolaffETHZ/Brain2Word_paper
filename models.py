import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dropout, Dense, LeakyReLU, Layer, BatchNormalization, Concatenate, Softmax
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2,l1_l2,l1
from tensorflow.keras import backend as K
from tensorflow.keras import activations


#Files needed: sizes and reduced sizes


class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name='bias',shape=self.dense.input_shape[-1],initializer='zeros')
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0],transpose_b=True)
        return self.activation(z+self.biases)


class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                shape=(self.embedding_dim, self.num_embeddings),
                                initializer=self.initializer,
                                trainable=True)

        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                    - 2 * K.dot(flat_inputs, self.w)
                    + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)
        # quantized = x - tf.stop_gradient(quantized-x)

        # Metrics.
        #avg_probs = K.mean(encodings, axis=0)
        #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
        return quantized
    
    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)


def autoencoder(trainable, mean):
    #parameters:
    rate = 0.4
    dense_size = 200 
    glove_size = 300
    fMRI_size =  65730 
    reduced_size =  3221 
    gordon_areas = 333
    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')

    index1 = 0
    index = 0

    input_voxel = Input(shape=(fMRI_size,))

    # small ROI dense layers: Each ROI region has its own dense layer. The outputs are then concatenated and used in further layers

    branch_outputs = []
    dense_layers = []
    for i in range(gordon_areas):
        new_index = index + sizes[i]
        small_input = Lambda(lambda x: x[:,index:new_index],output_shape=(sizes[i],))(input_voxel)
        dense_layers.append(Dense(reduced[i]))
        small_out = dense_layers[i](small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs.append(small_out)
        index = new_index
    Concat = Concatenate()(branch_outputs)
    dense1 = BatchNormalization()(Concat)
    dense1 = Dropout(rate=rate)(dense1)

    #intermediate Layer: Reduce the output from the ROI small dense layer further. 
    # The output from this layer is also used for the autoencoder to reconstruct the fMRIs

    dense5 = Dense(dense_size)     
    out_furtherr = dense5(dense1)
    out_further = LeakyReLU(alpha=0.3)(out_furtherr)
    out_further = BatchNormalization()(out_further)
    out_further = Dropout(rate=rate)(out_further)

    #Glove layer: The output of this layer should represent the matching glove embeddings for their respective fMRI. 
    # A loss is only applied if the glove prediction model is run.
    
    dense_glove = Dense(300, trainable=trainable)
    out_glove = dense_glove(out_further)
    out_gloverr = LeakyReLU(alpha=0.3)(out_glove)
    out_gloverr = BatchNormalization()(out_gloverr)
    out_gloverr = Dropout(rate=rate)(out_gloverr)

    #Classification layer: It returns a proability vector for a given fMRI belonging to a certainword out of the possible 180 words. 
    # The loss is only calculated if the classification model is run.

    out_mid = Dense(180, activation='softmax', trainable=trainable, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005))(out_gloverr)     
    
    dense4 = DenseTranspose(dense5)(out_further)
    dense4 = LeakyReLU(alpha=0.3)(dense4)
    dense4 = BatchNormalization()(dense4)
    dense4 = Dropout(rate=rate)(dense4)

    branch_outputs1 = []
    for j in range(gordon_areas):
        new_index1 = index1+reduced[j] 
        small_input = Lambda(lambda x: x[:,index1:new_index1], output_shape=(reduced[j],))(dense4) 
        small_out = DenseTranspose(dense_layers[j])(small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs1.append(small_out)
        index1 = new_index1
    out = Concatenate()(branch_outputs1)


    Concat_layer = Lambda(lambda t: t ,name = 'concat') (Concat)
    Dense_layer = Lambda(lambda t: t ,name = 'dense_mid') (out_furtherr) 
    pred_class = Lambda(lambda t: t ,name = 'pred_class') (out_mid)
    pred_glove = Lambda(lambda t: t ,name = 'pred_glove') (out_glove)
    fMRI_rec = Lambda(lambda t: t, name='fMRI_rec')(out)

    if not mean:
        model= Model(inputs=[input_voxel],outputs=[fMRI_rec, pred_glove, pred_class])
    else:
        model = Model(inputs=[input_voxel],outputs=[fMRI_rec, pred_glove, pred_class, Concat_layer, Dense_layer])

    return model


def encdec_VQVAE():

    #parameters:
    rate = 0.4
    dense_size = 1000
    embedding_dim=300
    num_embeddings = 180
    commitment_cost = 0.25
    glove_size = 300
    fMRI_size = 65730
    reduced_size =  3221 
    gordon_areas = 333
    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')
    index1 = 0
    index = 0

    # small ROI dense layers: Each ROI region has its own dense layer. The outputs are then concatenated and used in further layers

    input_voxel = Input(shape=(fMRI_size,))
    branch_outputs = []
    for i in range(gordon_areas):
        new_index = index + sizes[i]
        small_input = Lambda(lambda x: x[:,index:new_index],output_shape=(sizes[i],))(input_voxel)
        small_out = Dense(reduced[i])(small_input)
        small_out = BatchNormalization()(small_out)
        branch_outputs.append(small_out)
        index = new_index


    dense1 = Concatenate()(branch_outputs)
    dense1 = LeakyReLU(alpha=0.3)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(rate=rate)(dense1)

    #intermediate Layer: Reduce the output from the ROI small dense layer further. 
    # The output from this layer is also used for the autoencoder to reconstruct the fMRIs

    
    dense5 = Dense(dense_size)
    out_further = dense5(dense1)
    out_further = LeakyReLU(alpha=0.3)(out_further)
    out_further = BatchNormalization()(out_further)
    out_further = Dropout(rate=rate)(out_further)

    #VQ-VAE layer and Classification layer: It returns a proability vector for a given fMRI belonging to a certainword out of the possible 180 words. 
    # These layers hve each one loss: One for the classification and one for the VQ-VAE layer output


    out_mid = Dense(glove_size)(out_further)  
    enc_inputs = out_mid

    enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc_inputs)
    out_mid = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)

    out_class = Dense(180, activation='softmax')(out_mid)     

    dense4 = DenseTranspose(dense5)(out_further)
    dense4 = LeakyReLU(alpha=0.3)(dense4) 
    dense4 = BatchNormalization()(dense4)
    dense4 = Dropout(rate=rate)(dense4)

    branch_outputs1 = []
    for j in range(gordon_areas):
        new_index1 = index1+reduced[j] 
        small_input = Lambda(lambda x: x[:,index1:new_index1], output_shape=(reduced[j],))(dense4) 
        small_out = Dense(sizes[j])(small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs1.append(small_out)
        index1 = new_index1
    out = Concatenate()(branch_outputs1)

    pred_glove = Lambda(lambda t: t ,name = 'pred_glove') (out_mid)
    fMRI_rec = Lambda(lambda t: t, name='fMRI_rec')(out)
    pred_class = Lambda(lambda t: t, name='pred_class')(out_class)

    model = Model(inputs=[input_voxel],outputs=[fMRI_rec, pred_glove, pred_class])
    return model, enc, enc_inputs



def encdec_big_model(concating):
    #parameters:
    rate = 0.4
    glove_size = 300
    fMRI_size = 65730
    gordon_areas = 333
    class_size = 180

    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')
    index1 = 0
    index = 0

    # small ROI dense layers: Each ROI region has its own dense layer. The outputs are then concatenated and used in further layers

    input_voxel = Input(shape=(fMRI_size,))
    branch_outputs = []
    dense_layers = []
    for i in range(gordon_areas):
        new_index = index + sizes[i]
        small_input = Lambda(lambda x: x[:,index:new_index],output_shape=(sizes[i],))(input_voxel)
        dense_layers.append(Dense(reduced[i]))
        small_out = dense_layers[i](small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs.append(small_out)
        index = new_index
    
    Con = Concatenate()(branch_outputs)
    dense1 = Dropout(rate=rate)(Con)

    #Glove layer: The output of this layer should represent the matching glove embeddings for their respective fMRI. 
    # A loss is only applied if the glove prediction model is run.
    
    dense5 = Dense(glove_size)
    out_glove = dense5(dense1)
    # out_glove = LeakyReLU(alpha=0.3)(out_glove)
    # out_glove = BatchNormalization()(out_glove)

    #Classification layer: It returns a proability vector for a given fMRI belonging to a certainword out of the possible 180 words. 
    # The loss is only calculated if the classification model is run.

    dense6 = Dense(class_size)
    out_further = dense6(dense1)
    out_class = Softmax()(out_further)

    Concater = Lambda(lambda t: t ,name = 'Concat') (Con)
    pred_glove = Lambda(lambda t: t ,name = 'pred_glove') (out_glove)
    pred_class = Lambda(lambda t: t, name='pred_class')(out_class)

    if concating:
        model = Model(inputs=[input_voxel],outputs=[pred_glove, pred_class, Concater])
    else:
        model = Model(inputs=[input_voxel],outputs=[pred_glove, pred_class])
    return model


def encdec_small_model():
    #parameters:
    rate = 0.4
    glove_size = 300
    dense_size = 2000
    fMRI_size = 65730 #61656
    gordon_areas = 333
    class_size = 180

    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')
    index1 = 0
    index = 0

    input_voxel = Input(shape=(fMRI_size,))


    dense_first = Dense(dense_size)
    dense1 = dense_first(input_voxel)
    Con = LeakyReLU(alpha=0.3)(dense1)
    dense1 = BatchNormalization()(Con)
    dense1 = Dropout(rate=rate)(dense1)

    
    dense5 = Dense(glove_size)
    out_glove = dense5(dense1)
    # out_glove = LeakyReLU(alpha=0.3)(out_glove)
    # out_glove = BatchNormalization()(out_glove)

    dense6 = Dense(class_size)
    out_class = dense6(dense1)
    out_class = Softmax()(out_class)


    pred_glove = Lambda(lambda t: t ,name = 'pred_glove') (out_glove)
    pred_class = Lambda(lambda t: t, name='pred_class')(out_class)

    model = Model(inputs=[input_voxel],outputs=[pred_glove, pred_class])
    return model