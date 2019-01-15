import pandas as pd
import jieba.posseg as pseg
import keras
import numpy as np 
from sklearn.model_selection import train_test_split
from keras import layers

print('----------------------------------------------------------Print-------------------------------------------------------')

def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([word for word, flag in words if flag != 'x'])

train = pd.read_csv('/home/au/文件/Data/NLP Data/train.csv', index_col=0, encoding = 'utf-8')
train = train.astype(str)
cols = ['title1_zh', 'title2_zh', 'label']
train = train.loc[0:100, cols]

train['title1_tokenized'] =train.loc[0:100, 'title1_zh'].apply(jieba_tokenizer)
train['title2_tokenized'] =train.loc[0:100, 'title2_zh'].apply(jieba_tokenizer)

# print("======================================")
# print(train.iloc[0:, [0, 3]].head())
# print(train.iloc[0:, [1, 4]].head())


corpus_x1 = train.title1_tokenized
corpus_x2 = train.title2_tokenized
corpus = pd.concat([corpus_x1, corpus_x2])
# print(corpus.shape)
print(pd.DataFrame(corpus.iloc[:5], columns=['title']))
print("-" * 100)

MAX_NUM_WORDS = 10000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(corpus)
x1_train = tokenizer.texts_to_sequences(corpus_x1)
x2_train = tokenizer.texts_to_sequences(corpus_x2)

print(len(x1_train))
print(x1_train[:1])
print("-" * 100)


for seq in x1_train[:1]:
    print([tokenizer.index_word[idx] for idx in seq])

print("-" * 100)

for seq in x1_train[:10]:
    print(len(seq), seq[:5], ' ...')

print("-" * 100)

max_seq_len = max([
    len(seq) for seq in x1_train])

print(max_seq_len)
print("-" * 100)

MAX_SEQUENCE_LENGTH = 20
x1_train = keras.preprocessing.sequence.pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
x2_train = keras.preprocessing.sequence.pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)

print(x1_train[0])
print("-" * 100)

for seq in x1_train + x2_train:
    assert len(seq) == 20
    
print("所有新聞標題的序列長度皆為 20 !")
print("-" * 100)
print(x1_train[:5])
print("-" * 100)

# 定義每一個分類對應到的索引數字
label_to_index = {
    'unrelated': 0, 
    'agreed': 1, 
    'disagreed': 2
}

# 將分類標籤對應到剛定義的數字
y_train = train.label.apply(lambda x: label_to_index[x])
y_train = np.asarray(y_train).astype('float32')
print(y_train[:5])
print("-" * 100)

y_train = keras.utils.to_categorical(y_train)
print(y_train[:5])
print("-" * 100)

VALIDATION_RATIO = 0.1
RANDOM_STATE = 60

x1_train, x1_val, x2_train, x2_val, y_train, y_val =train_test_split(
   x1_train, x2_train, y_train, 
   test_size=VALIDATION_RATIO, 
   random_state=RANDOM_STATE
)



print("Training Set")
print("-" * 100)
print(f"x1_train: {x1_train.shape}")
print(f"x2_train: {x2_train.shape}")
print(f"y_train : {y_train.shape}")

print("-" * 100)
print(f"x1_val:   {x1_val.shape}")
print(f"x2_val:   {x2_val.shape}")
print(f"y_val :   {y_val.shape}")
print("-" * 100)
print("Test Set")

for i, seq in enumerate(x1_train[:5]):
    print(f"新聞標題 {i + 1}: ")
    print(seq)
    print()
print("-" * 100)

print(x1_train.shape)
print("-" * 100)

for i, seq in enumerate(x1_train[:5]):
    print(f"新聞標題 {i + 1}: ")
    print([tokenizer.index_word.get(idx, '') for idx in seq])
    print()
print("-" * 100)

# 基本參數設置，有幾個分類
NUM_CLASSES = 3

# 在語料庫裡有多少詞彙
MAX_NUM_WORDS = 10000

# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20

# 一個詞向量的維度
NUM_EMBEDDING_DIM = 256

# LSTM 輸出的向量維度
NUM_LSTM_UNITS = 128

# 建立孿生 LSTM 架構（Siamese LSTM）
from keras import Input
from keras.layers import Embedding, \
    LSTM, concatenate, Dense
from keras.models import Model

# 分別定義 2 個新聞標題 A & B 為模型輸入
# 兩個標題都是一個長度為 20 的數字序列
top_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')
bm_input = Input(
    shape=(MAX_SEQUENCE_LENGTH, ), 
    dtype='int32')

# 詞嵌入層
# 經過詞嵌入層的轉換，兩個新聞標題都變成
# 一個詞向量的序列，而每個詞向量的維度
# 為 256
embedding_layer = Embedding(
    MAX_NUM_WORDS, NUM_EMBEDDING_DIM)
top_embedded = embedding_layer(
    top_input)
bm_embedded = embedding_layer(
    bm_input)

# LSTM 層
# 兩個新聞標題經過此層後
# 為一個 128 維度向量
shared_lstm = LSTM(NUM_LSTM_UNITS)
top_output = shared_lstm(top_embedded)
bm_output = shared_lstm(bm_embedded)

# 串接層將兩個新聞標題的結果串接單一向量
# 方便跟全連結層相連
merged = concatenate(
    [top_output, bm_output], 
    axis=-1)

# 全連接層搭配 Softmax Activation
# 可以回傳 3 個成對標題
# 屬於各類別的可能機率
dense =  Dense(
    units=NUM_CLASSES, 
    activation='softmax')
predictions = dense(merged)

# 我們的模型就是將數字序列的輸入，轉換
# 成 3 個分類的機率的所有步驟 / 層的總和
model = Model(
    inputs=[top_input, bm_input], 
    outputs=predictions)

from keras.utils import plot_model
plot_model(
    model, 
    to_file='model.png', 
    show_shapes=True, 
    show_layer_names=False, 
    rankdir='LR')