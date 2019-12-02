
# coding: utf-8

# In[1]:


# 파이썬 확장 라이브러리 NLTK(Natural Language ToolKit) 설치 필요
# 설치 명령어 : pip3 install nltk 

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

porter = PorterStemmer()
stop = stopwords.words('english')

# 공백으로 단어 분리
def tokenizer(text):
    return text.split()

# Porter Stemming 알고리즘을 이용해 단어 분리
# 1979년 Martin F. Porter가 개발한 단어줄기(word stemming) 기법을 말함.
# -> 단어가 있을 때 그 단어의 원형으로 대체하여 분리하는 방법
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# In[ ]:




