#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from keras import backend
backend.set_image_dim_ordering('th')# set the image dim order of theano
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding,GRU,TimeDistributed,RepeatVector,Merge
from keras.layers import Embedding
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Input, Dense, merge
from keras.models import Model

from VGG import VGG_16
def videoCaption():
    '''

    :return: model with input shape [(None,5,3,224,224)(None,max_caption_len-1)] output shape （None，max_caption_len,10000）
    '''
    max_caption_len=21
    vocab_size=10000
    print "VGG loading"
    VGG_model = VGG_16()
    VGG_model.trainable = False

    input_sequences = Input(shape=(5, 3, 224, 224))
    processed_sequences = TimeDistributed(VGG_model)(input_sequences)  # outputshape=(None, 5, 4096)
    processed_sequences = GRU(output_dim=4096, return_sequences=False)(processed_sequences)
    processed_sequences = RepeatVector(1)(processed_sequences)  # (None, 1, 4096)

    print "Text model loading"
    input_sequences_0 = Input(shape=(max_caption_len - 1,), dtype='int32')
    input_sequences_l = Embedding(input_dim=vocab_size, output_dim=256, input_length=max_caption_len - 1)(
        input_sequences_0)  # input_shape==(None,max_caption_len),output_shape==(None,max_caption_len,256)
    input_sequences_l = GRU(output_dim=128, return_sequences=True)(
        input_sequences_l)  # input_shape==(None,max_caption_len-1,256),output_shape==(None,max_caption_len,128)
    input_sequences_l = TimeDistributed(Dense(4096))(input_sequences_l)  # (None,max_caption_len-1,4096)
    print "Merging"

    merged = merge([processed_sequences, input_sequences_l], mode='concat', concat_axis=1)
    # print VGG_model.output_shape
    output = GRU(256, return_sequences=True)(merged)  # output_shape=(None, 22, 256)
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    output = TimeDistributed(Dense(vocab_size))(output)
    output = Activation('softmax')(output)
    model = Model(input=[input_sequences, input_sequences_0], output=output)
    print "model compile"

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
if __name__ == "__main__":
    a=videoCaption()
    print a.input_shape
    print a.output_shape