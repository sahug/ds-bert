### **HuggingFace**: https://huggingface.co/ 

## **Steps**

- Load Data
- Split data to Train and Test
- Preprocess Data using Tensorflow Hub
- Encode data using Tensorflow Hub to input to BERT
- Build, Compile and Train model

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

- **input_word_ids** are the indices corresponding to each token in the sentence.
- **input_mask** indicates whether a token should be attended to or not.
- **input_type_ids** identifies which sequence a token belongs to when there is more than one sequence.

Calling **preprocess()** like this transforms raw text inputs into a fixed-length input sequence for the BERT encoder. You can see that it consists of a tensor input_word_ids with numerical ids for each tokenized input, including start, end and padding tokens, plus two auxiliary tensors: an input_mask (that tells non-padding from padding tokens) and input_type_ids for each token (that can distinguish multiple text segments per input, which we will discuss below).

### **2. BERT Pre-processing Model**

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

- **input_ids** are the indices corresponding to each token in the sentence.
- **attention_mask** indicates whether a token should be attended to or not.
- **token_type_ids** identifies which sequence a token belongs to when there is more than one sequence.

**Note:**
- The above preprocessing is for Text data. There are different preprocessing steps for Image and Audio data. You can check here: https://huggingface.co/docs/transformers/preprocessing
- You can use preprocessing from Tensorflow Hub or from Huggingface BERT. Make sure the data is compatible with the Encoder or Pre Trained Model.

## **Encoder**
When using the Tensorflow with BERT we need Encoder. The Encoder's outputs are the **pooled_output** to represents each input sequence as a whole, and the **sequence_output** to represent each input token in context. Either of those can be used as input to further model building.

The **output** of the encoder **pooled_output** or **sequence_output** will be an input to the BERT Model. 
The **input** to the Encoder is the Preprocessed Data.

```
encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
or
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

input = encoder(preprocess(["This is an amazing movie!"]))

```
**Note:** *Encoder is a pre trained model that we fine tune on our test data.*


**Table**
|Project|Coverage|
|-------|--------|
|BERT NLP - Introduction - BERT Machine Learning|Ktrain Text, Text Classifier, Ktrain Learner, DistilBert|
|BERT NLP - Sentiment Classification Using BERT|Ktrain Text, Text Classifier, Ktrain Learner, Bert|
|BERT NLP - Text Classification Using BERT|Tensorflow Text, Tensorflow Hub, KerasLayer, Bert Preprocess, Bert Encoder, Cosine Similarity, Functional API|
|BERT NLP - Multi Class Classification Using BERT and Tensorflow|Tensorflow, BertTokenizer, Tensorflow Dataset, TFAutoModel, 2 Input Layers, Save Model |
|BERT NLP - Intent Recognition Using BERT and Tensorflow|Tensorflow Hub, KerasLayer, Functional API|
|BERT NLP - Disaster Tweets using BERT and Tensorflow|Tensorflow Hub, Tensorflow Text, KerasLayer, Bert Preprocess, Bert Encoder|
|BERT NLP - Step by Step BERT|Tensorflow, ModelCheckpoint, TFAutoModel, AutoTokenizer, TQDM, Tokenizers, AUTOTUNE, Tensorflow Dataset|
|BERT NLP - Custom Training Q&A using BERT|QuestionAnsweringModel, QuestionAnsweringArgs|
|BERT NLP - Q&A using BERT|BertForQuestionAnswering, BertTokenizer, PyTorch|
|BERT NLP - IMDB - Text Classification using BERT|DistilBERT, Tensorflow, Datasets, AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, Create_Optimizer|
|BERT NLP - Named Entity Recognition or Token Classification using BERT|AutoTokenizer, Tensorflow, DistilBERT, DataCollatorForTokenClassification, Create_Optimizer, TFAutoModelForTokenClassification|
|BERT NLP - Question Answering using BERT|AutoTokenizer, Tensorflow, DistilBERT, DefaultDataCollator, Create_Optimizer, TFAutoModelForQuestionAnswering|
|BERT NLP - Sentiment140 - Sentiment Analysis using Tensorflow Hub and BERT|Tensorflow Hub, KerasLayer, Functional API|
