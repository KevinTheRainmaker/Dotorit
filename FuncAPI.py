import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import(Input, Softmax, Dense, Bidirectional, LSTM, Concatenate, concatenate, Embedding, Dropout, BatchNormalization)


EMBEDDING_DIM=256
MAXLEN = 32
TRUNCATING = 'post'
PADDING = 'pre'
OOV_TOKEN = '<OOV>'
NUM_WORDS = 15000


class MLP_Block(tf.keras.layers.Layer):
  def __init__(self, output_dim, drop_rate=0.3, name='block', loc='encoder', activation='relu', **kwargs):
    super(MLP_Block, self).__init__(name=name)
    self.FC = Dense(output_dim, activation=activation)
    self.BN = BatchNormalization()
    self.DP = Dropout(drop_rate)
    self.kwargs = kwargs
    self.loc = loc

    if self.loc=='decoder':
        self.Out = Dense(kwargs['final_dim'], activation=kwargs['final_activation'])
  
  #input : (1000, 1024)
  #output : (1000, 256)
  def call(self, inputs):
    x = self.FC(inputs)
    x = self.BN(x)
    x = self.DP(x)

    if self.loc == 'decoder':
      x = self.Out(x)

    return x


Text_processing = MLP_Block(output_dim = 512, name='text_layer')
Image_processing = MLP_Block(output_dim = 512, name='image_layer')
Feature_processing = MLP_Block(output_dim = 256, name = 'feature_layer')
Output_category = MLP_Block(output_dim = 128, name='category_output_layer', loc='decoder', final_dim=20, final_activation='softmax')
Output_price = MLP_Block(output_dim = 128, name='price_output_layer', loc='decoder', final_dim=1, final_activation='sigmoid')


input_text = tf.keras.Input(shape=[MAXLEN, NUM_WORDS])
input_image = tf.keras.Input(shape=[2048])

text = Embedding(NUM_WORDS, EMBEDDING_DIM, input_length=MAXLEN)(input_text)
text = Bidirectional(LSTM(16, return_sequences=False))(text[1,:,:,:])

txt = Text_processing(text)
img = Image_processing(input_image)


x = Concatenate(axis=-1)([txt,img])
x = Feature_processing(x)

out_category = Output_category(x)
out_price = Output_price(x)

model = tf.keras.Model(inputs=[input_text, input_image], outputs=[out_category, out_price])

model.summary()
