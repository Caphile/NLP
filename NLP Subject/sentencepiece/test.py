import sentencepiece as spm 

vocab_file = open('naver_review.txt', 'r', encoding='utf-8')
f = vocab_file.readlines()
f = f[9001:9011]

#spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

sp = spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)


for c, i in enumerate(f):
    print(str(c + 9001) + ' :', sp.encode_as_pieces(i))
    print('')


    print('')