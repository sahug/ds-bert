### **HuggingFace**:      https://huggingface.co/ 
### **Tensorflow BERT**:  https://blog.tensorflow.org/search?label=BERT
### **Tensorflow Hub**:   https://tfhub.dev/

## **Steps**

- Load Data
- Split data to Train and Test
- Preprocess Data using Tensorflow Hub
- Encode data using Tensorflow Hub to input to BERT
- Build, Compile and Train model

## **Token and IDs**
| Token | Meaning | Token ID |
| --- | --- | --- |
| **[PAD]** | Padding token, allows us to maintain same-length sequences (512 tokens for Bert) even when different sized sentences are fed in | 0 |
| **[UNK]** | Used when a word is unknown to Bert | 100 |
| **[CLS]** | Appears at the start of every sequence | 101 |
| **[SEP]** | Indicates a seperator - used to indicate point between context-question and appears at end of sequences | 102 |
| **[MASK]** | Used when masking tokens, for example in training with masked language modelling (MLM) | 103 |

## **Preprocess**

The standard or conventional procedure of pre-processing is a little bit tedious and also a user-centric procedure. The below steps are carried out under the hood of standard pre-processing techniques:

- Lower casing the corpus 
- Removing the punctuation 
- Removing the stopwords 
- Tokenizing the corpus 
- Stemming and Lemmatization
- Word embeddings using CountVectorizer and TF-IDF  

Nowadays all these pre-processing steps can be carried out by using transfer learning modules like BERT. BERT does this using the pre trained model. So we don't have to perform each of these tasks individually. Half of BERT’s success can be attributed to this pre-training phase. 

There are 2 ways we can pre-process our data.

### **1. Tensorflow Hub:**

TensorFlow Hub offers a variety of BERT and BERT-like models. For each BERT encoder, there is a matching preprocessing model. It transforms raw text to the numeric input tensors expected by the encoder, using TensorFlow ops provided by the **TF.text** library. Unlike preprocessing with pure Python, these ops can become part of a TensorFlow model for serving directly from text inputs. Each preprocessing model from TF Hub is already configured with a vocabulary and its associated text normalization logic and needs no further set-up.

```
import tensorflow_hub as hub
import tensorflow_text as text

preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')
or
preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")

input = preprocess(["This is an amazing movie!"])

```
```
{'input_word_ids': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
  array([[ 101, 2023, 2003, 2019, 6429, 3185,  999,  102,    0,  ...]])>,
 'input_mask': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
  array([[   1,    1,    1,    1,    1,    1,    1,    1,    0,  ...,]])>,
 'input_type_ids': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=
  array([[   0,    0,    0,    0,    0,    0,    0,    0,    0,  ...,]])>}
 ```
The tokenizer returns a dictionary with three important itmes:

- **input_word_ids:** Are the indices corresponding to each token in the sentence. Tensor of shape [batch_size, seq_length] with the token ids of the packed input sequence (that is, including a *start-of-sequence token, end-of-segment tokens, and padding*).
- **input_mask:** Indicates whether a token should be attended to or not. Tensor of shape [batch_size, seq_length] with value **1** at the position of all input tokens present before padding and value **0** for the padding tokens.
- **input_type_ids:** Identifies which sequence a token belongs to when there is more than one sequence. Tensor of shape [batch_size, seq_length] with the index of the input segment that gave rise to the input token at the respective position. The *first input segment (index 0) includes the start-of-sequence token and its end-of-segment token. The second and later segments (if present) include their respective end-of-segment token. Padding tokens get index 0 again.*

Calling **preprocess()** like this transforms raw text inputs into a fixed-length input sequence for the BERT encoder. You can see that it consists of a tensor **input_word_ids** with numerical ids for each tokenized input, including start, end and padding tokens, plus two auxiliary tensors: an **input_mask** (that tells non-padding from padding tokens) and **input_type_ids** for each token (that can distinguish multiple text segments per input, which we will discuss below).

### **2. BERT Pre-processing**

Before you can use your data in a model, the data needs to be processed into an acceptable format for the model. A model does not understand raw text, images or audio. These inputs need to be converted into numbers and assembled into tensors. In this tutorial, you will:

- Preprocess textual data with a tokenizer.
- Preprocess image or audio data with a feature extractor.
- Preprocess data for a multimodal task with a processor.

The main tool for processing textual data is a tokenizer. Load a pretrained tokenizer with **AutoTokenizer.from_pretrained()** or any other available BERT tokenizer.
```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")
```
- **padding=True**: Applies padding so that the length of each sentence is same.
- **truncation=True**: Sometimes a sequence may be too long for a model to handle. In this case, you will need to truncate the sequence to a shorter length.
- **return_tensors=tf**: Return the actual tensors that are fed to the model. **pt**: PyTorch or **tf**: Tensorflow.

Then pass your sentence to the tokenizer.

```
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)
```
```
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

The tokenizer returns a dictionary with three important itmes:

- **input_ids:** The input ids are often the only required parameters to be passed to the model as input. They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
- **attention_mask:** This argument indicates to the model which tokens should be attended to, and which should not. Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]. *1 for tokens that are not masked, 0 for tokens that are masked.*
- **token_type_ids:** These require two different sequences to be joined in a single “input_ids” entry, which usually is performed with the help of special tokens, such as the classifier ([CLS]) and separator ([SEP]) tokens. For example, the BERT model builds its two sequence input as such: `[CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]`

**Note:**
- The above preprocessing is for Text data. There are different preprocessing steps for Image and Audio data. You can check here: https://huggingface.co/docs/transformers/preprocessing
- You can use preprocessing from Tensorflow Hub or from Huggingface BERT. Make sure the data is compatible with the Encoder or Pre Trained Model.

## **Encoder or Pre trained Model**
When using the Tensorflow with BERT we need Encoder. The Encoder's outputs are the **pooled_output** to represents each input sequence as a whole, and the **sequence_output** to represent each input token in context. Either of those can be used as input to further model building.

The **output** of the encoder **pooled_output** or **sequence_output** will be an input to the BERT Model. 
The **input** to the Encoder is the Preprocessed Data.

```
encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
or
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

input = encoder(preprocess(["This is an amazing movie!"]))
```
- **sequence_output:** Tensor of shape [batch_size, seq_length, dim] with the context-aware embedding of each token of every packed input sequence.
- **pooled_output:** Tensor of shape [batch_size, dim] with the embedding of each input sequence as a whole, derived from sequence_output in some trainable manner.
- **default:** Required by the API for text embeddings with preprocessed inputs: a float32 Tensor of shape [batch_size, dim] with the embedding of each input sequence. (This might be just an alias of pooled_output.)

**Note:** *Encoder is a pre trained model that we fine tune on our test data.*

## **Finetune**
Once we have the data pre processed and we have the pre trained model we will train this model on our data. This phase is called Finetuneing. Here we have added additional layers, Dropout and Dense. You may need it ot you may not depending on what you want to achieve. You can just remove these layers and have the BERT layers only in the model.

```
import tensorflow as tf
from tensorflow import keras
from keras import activations

inputs = keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
preprocess = pre_processor(inputs)
encode = encode_input(preprocess)

nn1 = keras.layers.Dropout(0.1, name="dropout")(encode["pooled_output"])
nn1 = keras.layers.Dense(1, activation=keras.activations.sigmoid, name="output")(nn1)

model = keras.Model(inputs=[inputs], outputs=[nn1])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)
              
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2)              
```

### **Different Models**

**Note:** Input here are the columns that are feed to Tokenizer and passed down to encoder and the model. Output could be 1 or multiple based on model. While preparing the input we have to make sure we have the inputs as mentioned below. 

#### **Text Classification**

In this type of models we try to calssify the given text for example sentiment analysis or a movie review. In these cases we can have a tweet or a review written and we want to classify if the tweet or the review is good or bad or classify as any other sentiment. In these models we input the entire text, a tweet or review, as an input and its correspondent sentiment as output. Of course all input and outputs has to be pre processed, word embeding, label encoded to align with model requirement.

![image](https://user-images.githubusercontent.com/72315097/170854533-4cc670b8-d380-4401-aa66-55c55f14c1a1.png)

```
Example: BERT NLP - IMDB - Text Classification using BERT and Tensorflow

input: This is such a bad movie. I will never watch it again
output: bad

Example: preprocessor(data["text"])

**Note:** data has 1 column with text or a sentence.
```
#### **Named Entity Recognizion or Token Classification**
In this type of model we try to identify it a given word is a name, place, animal, fruit, etc. The input in these case will be a word and output will be a corresponding recognized entity.

![image](https://user-images.githubusercontent.com/72315097/170854609-1ee42e90-6686-4e15-9f86-181c52cc5e58.png)

**Word by Word**
```
Example: BERT NLP - Named Entity Recognition or Token Classification using BERT

input: James
output: name

preprocessor(data["word"))

O/P of the preprocessor is then feed to encoder or pre trained model

**Note:** data has 1 column with only 1 word.
```
**Sentence**
```
Example: BERT NLP - Named Entity Recognition or Token Classification using BERT - WNUT

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")
tf_train_set = tokenized_wnut["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)

```

#### **Question Answereing**
Depends on if the problem is a general question answering or a context based. In context based Q&A the answer is embedded in the context. 

The input for the general Q&A is a question and the context. We then build model and ask questions to get the ansers.

In context based Q&A we input question, context, start and end sequence of the answer embedded in the context. 

![image](https://user-images.githubusercontent.com/72315097/170854617-8740e744-cb3e-499b-b54a-18860df36bb1.png)

**General Q&A**
```
input:({question: "who is the richest man on earth?", context: "Elon Musk just passed Jeff Bezos to become the richest man on earth"})
output: "Elon Musk is the richest man on earth."

Example: preprocessor(data["question"], data["context"]) or preprocessor(data): 

**Note:** data has 2 columns, question and context.
```
**Context Based Q&A**
```
Example: BERT NLP - Question Answering using BERT and Tensorflow.

input:({question: "who is the richest man on earth?", context: "Elon Musk just passed Jeff Bezos to become the richest man on earth", start: 0, end: 9})
output: "Elon Musk"

Example: preprocessor(data["question"], data["context"], data["start"], data["end"]) or preprocessor(data): 

**Note:** data has 2 columns, question and context.
```

**Table**
|Project|Coverage|
|-------|--------|
|BERT NLP - Tokenizers|BertTokenizer, Tokenizer, Encode, Encode Plus, Batch Encode Plus, Special Tokens, Tensorflow Preprocessor|
|BERT NLP - Introduction - BERT Machine Learning|Ktrain Text, Text Classifier, Ktrain Learner, DistilBert|
|BERT NLP - Sentiment Classification Using BERT and Tensorflow|Ktrain Text, Text Classifier, Ktrain Learner, Bert|
|BERT NLP - Text Classification Using BERT and Tensorflow|Tensorflow Text, Tensorflow Hub, KerasLayer, Bert Preprocess, Bert Encoder, Cosine Similarity, Functional API|
|BERT NLP - Multi Class Classification Using BERT and Tensorflow|Tensorflow Hub, KerasLayer, Functional API|
|BERT NLP - Intent Recognition Using BERT and Tensorflow|Tensorflow Hub, KerasLayer, Functional API|
|BERT NLP - Disaster Tweets using BERT and Tensorflow|Tensorflow Hub, Tensorflow Text, KerasLayer, Bert Preprocess, Bert Encoder|
|BERT NLP - Step by Step BERT|Tensorflow, ModelCheckpoint, TFAutoModel, AutoTokenizer, TQDM, Tokenizers, AUTOTUNE, Tensorflow Dataset|
|BERT NLP - Custom Training Q&A using BERT|QuestionAnsweringModel, QuestionAnsweringArgs|
|BERT NLP - IMDB - Text Classification using BERT and Tensorflow|	Tensorflow Hub, KerasLayer, Functional API|
|BERT NLP - Named Entity Recognition or Token Classification using BERT - CONLL|Tensorflow Hub, KerasLayer, Functional API|
|BERT NLP - Named Entity Recognition or Token Classification using BERT - WNUT|AutoTokenizer, DataCollatorForTokenClassification, Create_Optimizer, TFAutoModelForTokenClassification|
|BERT NLP - Question Answering using BERT and Tensorflow|AutoTokenizer, Tensorflow, DistilBERT, DefaultDataCollator, Create_Optimizer, TFAutoModelForQuestionAnswering|
|BERT NLP - Sentiment140 - Sentiment Analysis using Tensorflow Hub and BERT|Tensorflow Hub, KerasLayer, Functional API|
