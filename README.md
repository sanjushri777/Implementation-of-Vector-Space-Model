
## NAME: SANJUSHRI A
## REGISTER NUMBER: 212223040187

## AIM:
To design and implement a Siamese Neural Network using LSTM to detect sentence similarity. The model uses cosine similarity to compute how similar two input sentences are and is trained on labeled sentence pairs.

## PROCEDURE


1. **Import Libraries**
   Load necessary libraries like TensorFlow, NumPy, and Keras modules.

2. **Create Dataset**
   Define sample sentence pairs with labels (1 = similar, 0 = not similar).

3. **Split Data**
   Separate the sentence pairs into two lists and extract their labels.

4. **Tokenization**
   Convert words into numerical tokens using `Tokenizer`.

5. **Sequence Padding**
   Pad the token sequences to ensure equal length for all inputs.

6. **Define Inputs**
   Create two input layers—one for each sentence in the pair.

7. **Embedding Layer**
   Add an embedding layer to convert tokens into dense vectors.

8. **Shared LSTM**
   Pass both inputs through the same Bidirectional LSTM layer.

9. **Similarity Calculation**
   Compute the L1 distance between the two encoded vectors and use a Dense layer to predict similarity.

10. **Train & Predict**
    Compile the model, train it with your dataset, and test it on new sentence pairs.



## PROGRAM:
```python


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Sentence pairs (labels: 1 for similar, 0 for dissimilar)
sentence_pairs = [
    ("How are you?", "How do you do?", 1),
    ("How are you?", "What is your name?", 0),
    ("What time is it?", "Can you tell me the time?", 1),
    ("What is your name?", "Tell me the time?", 0),
    ("Hello there!", "Hi!", 1),
]

# Separate into two sets of sentences and their labels
sentences1 = [pair[0] for pair in sentence_pairs]
sentences2 = [pair[1] for pair in sentence_pairs]
labels = np.array([pair[2] for pair in sentence_pairs])

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences1 + sentences2)
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
max_len = 100  # Max sequence length
X1 = pad_sequences(tokenizer.texts_to_sequences(sentences1), maxlen=max_len)
X2 = pad_sequences(tokenizer.texts_to_sequences(sentences2), maxlen=max_len)

# Input layers for two sentences
input_1 = Input(shape=(max_len,))
input_2 = Input(shape=(max_len,))

# Embedding layer
embedding_dim = 1000
embedding = Embedding(vocab_size, embedding_dim, input_length=max_len)

# Shared LSTM layer
shared_lstm = Bidirectional(LSTM(512))

# Process the two inputs using the shared LSTM
encoded_1 = shared_lstm(embedding(input_1))
encoded_2 = shared_lstm(embedding(input_2))

# Calculate the L1 distance between the two encoded sentences
def l1_distance(vectors):
    x, y = vectors
    return K.abs(x - y)

l1_layer = Lambda(l1_distance)
l1_distance_output = l1_layer([encoded_1, encoded_2])

# Add a dense layer for classification (similar/dissimilar)
output = Dense(1, activation='sigmoid')(l1_distance_output)

# Create the Siamese network model
siamese_network = Model([input_1, input_2], output)
siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
siamese_network.summary()

# Train the model
siamese_network.fit([X1, X2], labels, epochs=12, batch_size=2)

# Test with a new sentence pair
test_sentences1 = ["How are you?"]
test_sentences2 = ["How do you do?"]

test_X1 = pad_sequences(tokenizer.texts_to_sequences(test_sentences1), maxlen=max_len)
test_X2 = pad_sequences(tokenizer.texts_to_sequences(test_sentences2), maxlen=max_len)

# Predict similarity
similarity = siamese_network.predict([test_X1, test_X2])
print(f"Similarity Score: {similarity[0][0]}")

```

## OUTPUT:
![image](https://github.com/user-attachments/assets/fe3f981d-8d94-436d-aa94-2a2daa3ad260)

![image](https://github.com/user-attachments/assets/203d8313-d6dd-429b-aa01-5f01b09de71b)

## RESULT:
The Siamese LSTM network was successfully trained on sentence pairs.

