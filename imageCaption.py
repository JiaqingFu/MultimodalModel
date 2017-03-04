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
from VGG import VGG_16


def imageCaption():
    '''
    :return:model with input shape (None,3,100,100)(None,max_caption_len) output shape(None, max_caption_len, vocab_size)
    you can call it with beam search
    #reference:https://keras.io/getting-started/sequential-model-guide/#examples
    '''
    max_caption_len = 16
    vocab_size = 10000

    # first, let's define an image model that
    # will encode pictures into 128-dimensional vectors.
    # it should be initialized with pre-trained weights.
    image_model = Sequential()
    image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(32, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    image_model.add(Activation('relu'))
    image_model.add(Convolution2D(64, 3, 3))
    image_model.add(Activation('relu'))
    image_model.add(MaxPooling2D(pool_size=(2, 2)))

    image_model.add(Flatten())
    image_model.add(Dense(128))

    # let's load the weights from a save file.it is alternative
    #  image_model.load_weights('weight_file.h5')

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
    language_model.add(GRU(output_dim=128, return_sequences=True))
    language_model.add(TimeDistributed(Dense(128)))

    # let's repeat the image vector to turn it into a sequence.
    image_model.add(RepeatVector(max_caption_len))

    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
    # let's encode this vector sequence into a single vector
    model.add(GRU(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

def imageCaption2():
    '''

    :return: model with input shape (None,3,224,224)(None,max_caption_len) output shape(None, max_caption_len,vocab_size)
    you can call it with beam search
    replace the image model of imageCaption() with VGG_16
    '''
    max_caption_len = 21
    vocab_size = 10000
    print "VGG loading"
    #image_model = VGG_16('vgg16_weights.h5')
    image_model=VGG_16()
    image_model.trainable = False
    print "VGG loaded"
    # let's load the weights from a save file.
    # image_model.load_weights('weight_file.h5')

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    print "Text model loading"
    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))#input_shape==(None,max_caption_len),output_shape==(None,max_caption_len,256)
    language_model.add(GRU(output_dim=128, return_sequences=True))#input_shape==(None,max_caption_len,256),output_shape==(None,max_caption_len,128)
    language_model.add(TimeDistributed(Dense(128)))
    print "Text model loaded"
    # let's repeat the image vector to turn it into a sequence.
    print "Repeat model loading"
    image_model.add(RepeatVector(max_caption_len))#output_shape=(None,21,4096)
    print "Repeat model loaded"
    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    print "Merging"
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))#output_shape=(None, 21, 4224)
    # let's encode this vector sequence into a single vector
    model.add(GRU(256, return_sequences=False))
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
def imageCaption3():
    '''
    :return: model with input shape (None,3,224,224)(None,max_caption_len) output shape(None, max_caption_len+1,vocab_size)
    the difference between imageCaption2 and imageCaption3:the image vector of imageCaption3 is repeat once and take one position
    of RNN,so we get max_caption_len+1 in the output shape
     '''
    max_caption_len = 21
    vocab_size = 10000
    print "VGG loading"
    #image_model = VGG_16('vgg16_weights.h5')
    image_model = VGG_16()
    image_model.trainable = False
    print "VGG loaded"
    # let's load the weights from a save file.
    # image_model.load_weights('weight_file.h5')

    # next, let's define a RNN model that encodes sequences of words
    # into sequences of 128-dimensional word vectors.
    print "Text model loading"
    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256,
                                 input_length=max_caption_len - 1))  # input_shape==(None,max_caption_len),output_shape==(None,max_caption_len,256)
    language_model.add(GRU(output_dim=128,
                           return_sequences=True))  # input_shape==(None,max_caption_len,256),output_shape==(None,max_caption_len,128)
    language_model.add(TimeDistributed(Dense(4096)))
    print "Text model loaded"
    # let's repeat the image vector to turn it into a sequence.
    print "Repeat model loading"
    image_model.add(RepeatVector(1))  # output_shape=(None,1,4096)
    print "Repeat model loaded"
    # the output of both models will be tensors of shape (samples, max_caption_len, 128).
    # let's concatenate these 2 vector sequences.
    print "Merging"
    model = Sequential()
    model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))  # output_shape=(None, 22, 4096)
    # let's encode this vector sequence into a single vector
    model.add(GRU(256, return_sequences=True))  # output_shape=(None, 22, 256)
    # which will be used to compute a probability
    # distribution over what the next word in the caption should be!
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
if __name__ == "__main__":
    imageCaption3()
    imageCaption2()
    imageCaption()