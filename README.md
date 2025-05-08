# 代码核心功能说明
## 1.文本预处理与特征提取
### 文本预处理与分词过滤
**从文件中提取文本，去除无效字符，进行分词处理，并过滤掉长度为 1 的单词，最终返回一个清洗后的单词列表。**
```python
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤无效字符
            line = cut(line)  # jieba分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤长度为1的词
            words.extend(line)
    return words
```
### 构建高频词库
**get_top_words 函数从 151 个邮件文件中提取单词，统计词频并返回出现频率最高的 top_num 个单词。**
```python
def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))  # 遍历所有邮件生成词库
    freq = Counter(chain(*all_words))  # 统计词频
    return [i[0] for i in freq.most_common(top_num)]  # 返回前 top_num 个高频词
```
## 2.特征向量化
### 词频统计
**将 all_words 中每篇文档的特征词（top_words）词频统计为向量，最终转化为 NumPy 数组。**
```python
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))  # 统计每个特征词的词频
    vector.append(word_map)
vector = np.array(vector)  # 转换为 NumPy 数组
```
### 标签标记
**为训练数据分配标签，标记垃圾邮件和普通邮件。前127封邮件标记为垃圾邮件（1），后24封标记为普通邮件（0）。**
## 3.模型训练
### 分类算法
**使用多项式朴素贝叶斯模型对文本向量 vector 和标签 labels 进行训练。。**
```python
model = MultinomialNB()  # 初始化多项式朴素贝叶斯模型
model.fit(vector, labels)  # 使用词频向量和标签进行训练
```
## 4.新邮件分类
### 预测逻辑
**加载新邮件，提取词频向量，用模型预测并返回“垃圾邮件”或“普通邮件”。**
```python
def predict(filename):
    words = get_words(filename)  # 预处理新邮件
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))  # 生成词频向量
    result = model.predict(current_vector.reshape(1, -1))  # 预测结果
    return '垃圾邮件' if result == 1 else '普通邮件'
```
# 高频词/TF-IDF两种特征模式及其切换方法
## 高频词特征模式
### 使用词袋模型
**使用 CountVectorizer 对文本数据进行词频统计，并将其转换为数值向量。**
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
```
## TF-IDF特征模式
### 使用TF-IDF模型
**使用 CountVectorizer 对文本数据进行词频统计，并将其转换为数值向量。**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())
```
## 转换方法
### 从高频词切换到TF-IDF
**使用 `CountVectorizer` 将文本转换为词频矩阵，再通过 `TfidfTransformer` 转换为 TF-IDF 特征向量并输出。**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
print(tfidf.toarray())
```
### 从TF-IDF切换到高频词
**使用`CountVectorizer`将文本`corpus`转换为词频矩阵并输出数组形式。**
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["The quick brown fox jumps over the lazy dog.", "The quick brown fox is fast."]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```