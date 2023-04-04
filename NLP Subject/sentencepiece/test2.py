from konlpy.tag import Okt


okt = Okt()

vocab_file = open('naver_review.txt', 'r', encoding='utf-8')
f = vocab_file.readlines()

morphs = []
for sentence in f:
    morphs.append(okt.morphs(sentence))

for c, i in enumerate(morphs):
    print(str(c + 9001) + ' :', i)
    print('')

print('')