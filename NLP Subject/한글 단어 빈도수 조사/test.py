from konlpy.tag import Okt
from collections import Counter

stopword = "은 는 이 가 을 를 이 것 저 그 했다 었다"  # 불용어 리스트
stopword = set(stopword.split(' '))

def process(fileName, bulChk = False):

    print("파일 명     : " + fileName)
    if bulChk == True:
        chk = 'O'
    else:
        chk = 'X'
    print("불용어 제거 : " + chk + "\n")

    infile = open(fileName, encoding = 'utf-8')
    data = infile.read()

    okt = Okt()
    words = okt.nouns(data)
    if bulChk == True:  # 불용어 제거
        words = [word for word in words if not word in stopword]        

    vocab = Counter(words)

    num = vocab.values()
    sum = 0
    for i in num:
        if i >= 5:
            sum += 1

    vocab_2 = Counter(dict(vocab.most_common(sum)))

    print(vocab_2)
    print('-----------------------------------------------------------------------------------------------------')

process('삼포 가는 길.txt')
process('삼포 가는 길.txt', 1)
process('소나기.txt')
process('소나기.txt', 1)


