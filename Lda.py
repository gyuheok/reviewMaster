
# [NLP] 한국어 토픽 모델링 - LDA 활용방법(gensim)
import pandas as pd
from konlpy.tag import Okt
from gensim.corpora.dictionary import Dictionary

df = pd.read_csv('C:\\Users\\hangy\\Downloads\\petition_sampled.csv')
df.head()

#사용할 부분만 남기기
contents = df['content']
contents

df.groupby('category').votes.sum() #17개 카테고리

len(df.groupby('category').votes.sum())#17개

###형태소 분석###
okt = Okt()
def analysis_pos(text):
    morphs = okt.pos(text, stem=True)

    words = []
    #명사 추출, 2글자 이상 단어 추출
    for word, pos in morphs:
        if pos == 'Noun':
            if len(word) > 1:
                words.append(word)
    return words

texts = [analysis_pos(news) for news in contents]