sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
#分词
word_sequence = " ".join(sentences).split() # ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
print("word_sequence:",word_sequence)
vocab = list(set(word_sequence)) # build words vocabulary
print("vocab:",vocab)
word2idx = {w: i for i, w in enumerate(vocab)} # {'jack':0, 'like':1,...}
print("word2idx:",word2idx)
