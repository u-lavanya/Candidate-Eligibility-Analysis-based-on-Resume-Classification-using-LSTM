#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from tqdm import tqdm_notebook

from nltk.corpus import stopwords

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# In[2]:


MAX_NB_WORDS = 10000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100  


# In[4]:


import pandas as pd
train = pd.read_csv('/Users/ssmie/OneDrive/Desktop/nldata/Train.csv')
test = pd.read_csv('/Users/ssmie/OneDrive/Desktop/nldata/Test.csv')


# In[5]:


train.head()


# In[6]:


train.isnull().sum()


# In[7]:


test.isnull().sum()


# In[8]:


labels = ['Experience', 'Eligible', 'Not_Eligible']
y = train[labels].values
Technology_train = train['Technology']
Technology_test = test['Technology']


# In[9]:


Technology_train = list(Technology_train)


# In[10]:


def clean_text(text, remove_stopwords = True):
    output = ""
    text = str(text).replace("\n", "")
    text = re.sub(r'[^\w\s]','',text).lower()
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
    else:
        output = text
    return str(output.strip())[1:-3].replace("  ", " ")


# In[11]:


import nltk
nltk.download('popular')


# In[12]:


texts = [] 

for line in tqdm_notebook(Technology_train, total=159571): 
    texts.append(clean_text(line))


# In[13]:


print('Sample data:', texts[1], y[1])


# In[14]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)


# In[15]:


sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))


# In[16]:


data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)


# In[19]:


import numpy as np

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]


# In[69]:


num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])
x_train = data[: -num_validation_samples]
y_train = labels[: -num_validation_samples]
x_val = data[-num_validation_samples: ]
y_val = labels[-num_validation_samples: ]


# In[70]:


print('Number of entries in each category:')
print('training: ', y_train.sum(axis=0))
print('validation: ', y_val.sum(axis=0))


# In[71]:


print('Tokenized sentences: \n', data[10])
print('One hot label: \n', labels[10])


# In[81]:


EMBEDDING_DIM = 100
GLOVE_DIR = "/Users/ssmie/OneDrive/Desktop/nldata/glove.6B."+str(EMBEDDING_DIM)+"d.txt"


embeddings_index = {}
f = open(GLOVE_DIR,'rb')
print('Loading GloVe from:', GLOVE_DIR,'...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done.\n Proceeding with Embedding Matrix...", end="")

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed!")



# In[82]:


#weight = "/Users/ssmie/Downloads/my_model_weights_discriminator.h5"
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                           EMBEDDING_DIM,
                           weights = [embedding_matrix],
                           input_length = MAX_SEQUENCE_LENGTH,
                           trainable=False,
                           name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)


# In[83]:


x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(3, activation="sigmoid")(x)


# In[84]:


model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
model.summary()


# In[51]:


y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))


# In[ ]:





# In[49]:


#import tensorflow as tf
#tf.keras.utils.plot_model(model)


# In[86]:


print('Training progress:')
history = model.fit(x_train, y_train, epochs = 5, batch_size=32, validation_data=(x_val, y_val))


# In[88]:


import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show();


# In[89]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show();


# In[ ]:




