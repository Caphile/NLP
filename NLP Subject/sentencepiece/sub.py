import sentencepiece as spm 

num = 110500

vocab_file = open('naver_review.txt', 'r', encoding='utf-8')
f = vocab_file.readlines()
f = f[num:num + 11]

#spm.SentencePieceTrainer.Train('--input=sub.txt --model_prefix=sub --vocab_size=32000 --model_type=bpe --max_sentence_length=9999')

sp = spm.SentencePieceProcessor()
vocab_file = "sub.model"
sp.load(vocab_file)


for c, i in enumerate(f):
    print(str(c + num) + ' :', sp.encode_as_pieces(i))
    print('')


print('')