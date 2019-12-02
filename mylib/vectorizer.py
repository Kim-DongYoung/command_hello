
# coding: utf-8

# In[2]:


# 앞서 만든 csv 파일을 읽어서 pandas의 DataFrame 객체로
# 저장한 후, 특수기호, HTML 태그 등을 제거하고 모두 소문자로 변환하여
# 새로운 csv 파일로 저장하는 로직.
# 
# 위 함수에서 정지단어를 제거하여 리턴하는 로직을 추가한 함수

from sklearn.feature_extraction.text import HashingVectorizer
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')

def sgd_tokenizer(text):
    # 특수기호, HTML 태그 등 제거 (단, 이모티콘은 남겨둠)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=sgd_tokenizer)


# In[ ]:




