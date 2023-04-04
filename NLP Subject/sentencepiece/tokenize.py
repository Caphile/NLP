import pandas as pd 
import sentencepiece as spm 
import csv

#urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt") 


naver_df = pd.read_table('ratings.txt')
naver_df = naver_df.dropna(how = 'any')

with open('naver_review.txt', 'w', encoding='utf8') as f:
	f.write('\n'.join(naver_df['document']))


print('Line : ' + str(len(naver_df)))
print('Word : 5000')

vocab_list = pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)



print()