import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from underthesea import word_tokenize
import shap
from lime import lime_image
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(base_path='../dataset'):
    image_paths = []
    captions = []
    for img_name in os.listdir(f'{base_path}/images'):
        if img_name.endswith('.jpg'):
            image_path = f'{base_path}/images/{img_name}'
            caption_path = f'{base_path}/captions/{img_name.replace(".jpg", ".txt")}'

            with open(caption_path, 'r') as f:
                caption = f.read()

            image_paths.append(image_path)
            captions.append(caption)

    return image_paths, captions

image_paths, captions = load_dataset()
print(f'Loaded {len(image_paths)} images and {len(captions)} captions')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# PCA cho hình ảnh
pca = PCA(n_components=128) 

image_data = []
image_features = {}

for img_path in image_paths:
    preprocessed_img = preprocess_image(img_path)
    features = resnet.predict(preprocessed_img, verbose=0)
    image_data.append(features.flatten())

image_data = np.array(image_data)
image_data_pca = pca.fit_transform(image_data)

for i, img_path in enumerate(image_paths):
    image_id = img_path.split('/')[-1].split('.')[0]
    image_features[image_id] = image_data_pca[i]

images = list(image_features.keys())

# Tokenization và Padding cho tiếng Việt
tokenizer = Tokenizer(oov_token="<unk>")
captions_tokenized = [word_tokenize(caption) for caption in captions]
tokenizer.fit_on_texts(captions_tokenized)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(captions_tokenized)
max_length = max(len(s) for s in sequences)
captions_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

print(f'Vocab size: {vocab_size}, Max length: {max_length}')
# print samples
for i in range(3):
    print(f'Original: {captions[i]}')
    print(f'Tokenized: {captions_tokenized[i]}')
    print(f'Padded: {captions_padded[i]}')
    print()

def build_model(vocab_size, max_length):
    # Image feature extractor layer
    inputs1 = Input(shape=(2048,))  # ResNet50 output
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor layer
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder layer
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

vocab_size = len(tokenizer.word_index) + 1
model = build_model(vocab_size, max_length)
model.summary()

def data_generator(captions, image_features, tokenizer, max_length, batch_size):
    X1, X2, y = list(), list(), list()
    n=0
    while 1:
        for i, caption in enumerate(captions):
            n+=1
            image_id = images[i]
            photo = image_features[image_id]
            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)

            if n == batch_size:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0


