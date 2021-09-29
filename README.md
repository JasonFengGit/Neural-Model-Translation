# ZH-EN NMT Chinese to English Neural Machine Translation

> This project is inspired by Stanford's CS224N NMT Project
>
> Dataset used in this project: [News Commentary v14](http://data.statmt.org/news-commentary/v14)

## Intro

This project is more of a learning project to make myself familiar with Pytorch, machine translation, and NLP model training.

To investigate how would various setups of the recurrent layer affect the final performance, I compared Training Efficiency and Effectiveness of different types of RNN layer for encoder by changing one feature each time while controlling all other parameters:

- RNN types
  - GRU
  - LSTM
- Activation Functions on Output Layer
  - Tanh
  - ReLU
- Number of layers

  - single layer
  - double layer

## Code Files

```
_/
├─ utils.py # utilities
├─ vocab.py # generate vocab
├─ model_embeddings.py # embedding layer
├─ nmt_model.py # nmt model definition
├─ run.py # training and testing
```

## Good Translation Examples

- ***source***: 相反,这意味着合作的基础应当是共同的长期战略利益,而不是共同的价值观。
  - ***target***: Instead, it means that cooperation must be anchored not in shared values, but in shared long-term strategic interests.
  - ***translation***: On the contrary, that means cooperation should be a common long-term strategic interests, rather than shared values.

- ***source***: 但这个问题其实很简单: 谁来承受这些用以降低预算赤字的紧缩措施的冲击。
  - ***target***: But the issue is actually simple: Who will bear the brunt of measures to reduce the budget deficit?
  - ***translation***: But the question is simple: Who is to bear the impact of austerity measures to reduce budget deficits?
- ***source***: 上述合作对打击恐怖主义、贩卖人口和移民可能发挥至关重要的作用。
  - ***target***: Such cooperation is essential to combat terrorism, human trafficking, and migration.
  - ***translation***: Such cooperation is essential to fighting terrorism, trafficking, and migration.
- ***source***: 与此同时, 政治危机妨碍着政府追求艰难的改革。
  - ***target***: At the same time, political crisis is impeding the government’s pursuit of difficult reforms.
  - ***translation***: Meanwhile, political crises hamper the government’s pursuit of difficult reforms.

## Preprocessing

> Preprocessing Colab [notebook](https://colab.research.google.com/drive/1IJTdk7hj3uoPEE0Ox7QaeW4rTuUzuxPJ?usp=sharing)

- using [`jieba` ](https://github.com/fxsjy/jieba)to separate Chinese words by spaces

## Generate Vocab From Training Data

- Input: training data of Chinese and English

- Output: a vocab file containing mapping from (sub)words to ids of Chinese and English -- a limited size of vocab is selected using [SentencePiece](https://github.com/google/sentencepiece) (essentially [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) of character n-grams) to cover around 99.95% of training data

## Model Definition

- a Seq2Seq model with attention

  > This image is from the book [DIVE INTO DEEP LEARNING](https://zh-v2.d2l.ai/index.html)

  ![](https://zh-v2.d2l.ai/_images/seq2seq-attention-details.svg)

  - Encoder
    - A Recurrent Layer
  - Decoder
    - LSTMCell (hidden_size=512)
  - Attention
    - Multiplicative Attention

## Training And Testing Results

> Training Colab [notebook](https://colab.research.google.com/drive/1HYbOh0AUMEasBAH7QPGNq9joH2dRRZwg?usp=sharing)

- **Hyperparameters:**
  - Embedding Size & Hidden Size: 512
  - Dropout Rate: 0.25
  - Starting Learning Rate: 5e-4
  - Batch Size: 32
  - Beam Size for Beam Search: 10
- **NOTE:** The BLEU score calculated here is based on the `Test Set`, so it could only be used to compare the **relative effectiveness** of the models using this data

#### For Experiment

- **Dataset:** the dataset is split into training set(~260000), validation set(~20000), and testing set(~20000) randomly (they are the same for each experiment group)
- **Max Number of Iterations**: 50000
- **NOTE:** I've tried `Vanilla-RNN(nn.RNN)` in various ways, but the BLEU score turns out to be extremely low for it, I decided to not include it for comparison until the issue is resolved

|                                                  | Training Time(sec) | BLEU Score on Test Set | Training Perplexities                                        | Validation Perplexities                                      |
| ------------------------------------------------ | ------------------ | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **A.** Bidirectional 1-Layer GRU with Tanh       | 5158.99            | 14.26                  | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/edb3bca3d2c190398ab211195d9b14a16a163d76/images/gru_train_ppl.svg) | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/edb3bca3d2c190398ab211195d9b14a16a163d76/images/gru_dev_ppl.svg) |
| **B.** Bidirectional 1-Layer LSTM with Tanh      | 5150.31            | 16.20                  | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/edb3bca3d2c190398ab211195d9b14a16a163d76/images/lstm_train_ppl.svg) | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/edb3bca3d2c190398ab211195d9b14a16a163d76/images/lstm_dev_ppl.svg) |
| **C.** Bidirectional 2-Layer LSTM with Tanh      | 6197.58            | **16.38**              | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/4e70246b618a0fa35d5ab75193df638ac1e27562/images/lstm_2_layer_train_ppl.svg) | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/4e70246b618a0fa35d5ab75193df638ac1e27562/images/lstm_2_layer_dev_ppl.svg) |
| **D.** Bidirectional 1-Layer LSTM with ReLU      | 5275.12            | 14.01                  | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/4e70246b618a0fa35d5ab75193df638ac1e27562/images/lstm_relu_train_ppl.svg) | ![](https://raw.githubusercontent.com/JasonFengGit/ZH-EN-Neural-Model-Translation/4e70246b618a0fa35d5ab75193df638ac1e27562/images/lstm_relu_dev_ppl.svg) |
| **E.** Bidirectional 1-Layer LSTM with LeakyReLU | TODO               | TODO                   | TODO                                                         | TODO                                                         |

#### Best Version

TODO

#### Analysis

- LSTM tends to have better performance than GRU (it has an extra set of parameters)
- Tanh tends to be better since less information is lost
- Making the LSTM deeper (more layers) could improve the performance, but it cost more time to train
- Surprisingly, the training time for A, B, and D are roughly the same
  - the issue may be the dataset is not large enough, or the cloud service I used to train models does not perform consistently

## Bad Examples & Case Analysis

- ***source***: **全球目击组织(Global Witness)**的报告记录, 光是2015年就有**16个国家**的185人被杀。
  - ***target***: A **Global Witness** report documented 185 killings across **16 countries** in 2015 alone.
  - ***translation***: According to the **Global eye**, the World Health Organization reported that 185 people were killed in 2015.
  - ***problems***: 
    - Information Loss: 16 countries
    - Unknown Proper Noun: Global Witness
- ***source***: 大自然给了足以满足每个人需要的东西, **但无法满足每个人的贪婪**。
  - ***target***: Nature provides enough for everyone’s needs, **but not for everyone’s greed**.
  - ***translation***: Nature provides enough to satisfy everyone.
  - ***problems***: 
    - Huge Information Loss
- ***source***: 我衷心希望全球经济危机和巴拉克·奥巴马当选总统能对新冷战的荒唐理念进行正确的评估。
  - ***target***: It is my hope that the global economic crisis and Barack Obama’s presidency will put the farcical idea of a new Cold War into proper perspective.
  - ***translation***: I do hope that the global economic crisis and President Barack Obama will be corrected for a new Cold War.
  - ***problems***: 
    - Action Sender And Receiver Exchanged
    - Failed To Translate Complex Sentence
- ***source***: 人们纷纷**猜测**欧元区将崩溃。
  - ***target***: **Speculation** about a possible breakup was widespread.
  - ***translation***: The eurozone would collapse.
  - ***problems***: 
    - Significant Information Loss

## Means to Improve the NMT model

- Dataset	
  - The dataset is fairly small, and our model is not being trained thorough all data
  - Being a native Chinese speaker, I could not understand what some of the source sentences are saying
  - The target sentences are not informational comprehensive; they themselves need context to be understood (e.g. the target sentence in the last "Bad Examples")
  - Even for human, some of the source sentence was too hard to translate
- Model Architecture
  - CNN & Transformer
  - character based model
  - Make the model even larger & deeper (... I need GPUs)
- Tricks that might help
  - Add a proper noun dictionary to translate unknown proper nouns word-by-word (phrase-by-phrase) 
  - Initialize (sub)word embedding with pretrained embedding

## How To Run
- To run locally on a CPU (mostly for sanity check, CPU is not able to train the model)
  - set up the environment using conda/miniconda `conda env create --file local env.yml`
- To run on a GPU
  - set up the environment and running process following the Colab [notebook](https://colab.research.google.com/drive/1HYbOh0AUMEasBAH7QPGNq9joH2dRRZwg?usp=sharing)


## Contact
If you have any questions or you have trouble running the code, feel free to contact me via [email](mailto:jasonfen@usc.edu)

