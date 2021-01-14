import numpy as np
import nltk
from   nltk.corpus import stopwords
from   nltk.stem import PorterStemmer

def read_train_data(filename):
    with open(filename) as f:
        content = f.readlines()

    content = [x for x in content]
    
    return  content

def split_to_ner_sentences(content):
    sentences = []
    temp_sentence = []
    for word in content:
#        print(word)
        if word == "\n":
            temp_sentence = [x.strip("\n") for x in temp_sentence]
            sentences.append(temp_sentence)
            temp_sentence=[]
        else:
            testit = word.split(" ")
#            if "-X-" not in testit:
            if word != ""  and word != "-DOCSTART- -X- O O" and word!="-DOCSTART- -X- -X- O":
                temp_sentence.append(word)

    return sentences
        
def get_tokens(content):
    content = [x.strip("\n") for x in content]
#    content = [x.strip(" ") for x in content]

    temp_sentence = ""
    tokens=[]
    for item in content:
        words = item.split(" ")
        tokens.append(words[0]) #token is always  at zero-th position of WORD POS CHUNK NamedEntity
#    tokens = nltk.word_tokenize(temp_sentence)
#    tokens_without_sw = [word for word in tokens if not word in stopwords.words()]
    return tokens

def get_pos_tags(content):
    pos = []
    for item in content:
        w = item.split(" ")
        pos.append(w[1])
    return pos
    
def get_chunks(content):
    chunks = []
    for item in content:
        w = item.split(" ")
        chunks.append(w[2])
    return chunks
    
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    tok=[]
    for t in tokens:
        tok.append(stemmer.stem(t))
    return tok

def get_vocabulary_as_dict(tokens):
    vocabulary = {}
    c=0
    idx=0
#    print("len", len(tokens))
    for idx,t in enumerate(tokens):
#        if t not in vocabulary:
        vocabulary[t]=idx
        c=idx
#    print("c",c)
    c+=1
    vocabulary['OOV']=c
    return vocabulary

def get_prexif_as_dict(tokens, position):
    pref = {}
    count = 0#(start from minimum idx 1 so thatI can keep 0  for words shorter than 2,3  tokens)
    for idx,t in enumerate(tokens):
        if len(t) > position: # only extract prefix if token is long enough
            if t[position] not in pref.keys():
                pref[t[position]]=count
                count+=1
    pref['NOCHAR'] = count
    return pref
    
def glove_embeddings(glove_filename):
    embeddings_dict = {}
    mean_vec  = np.zeros(50)
    c = 0
    with open(glove_filename, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            mean_vec += vector
            embeddings_dict[word] = vector
            c+=1
    return embeddings_dict, mean_vec/c
    
    
def tokens_from_glove(tokens, glove, mean_vec):
    tok = {}
    for t in tokens:
        if t in glove:
            tok[t] = glove[t]
        else:
            tok[t] = mean_vec
    return tok

def get_labels(content, one_hot):
    labels = []
    for item in content:
        w = item.split(" ")
        labels.append(one_hot[w[3]])
    return labels
    
def labels_to_onehot_dictionary(raw_data_filename):
    sentences = split_to_ner_sentences(read_train_data(raw_data_filename))

    labels = {}
    for s in sentences:
        for item in s:
            w = item.split(" ")
            labels[w[3]] = 0
    
    num_of_labels =  len(labels.keys())
    one_hot       =  np.zeros(num_of_labels)
    count = 0
    for key in labels:
        enc         = np.zeros(num_of_labels).tolist()
        enc[count]  = 1
        labels[key] = count
        count+=1

    return labels
  
def pos_to_onehot_dictionary(raw_data_filename):
    sentences = split_to_ner_sentences(read_train_data(raw_data_filename))
    pos = {}
    for s in sentences:
        for item in s:
            w = item.split(" ")
            pos[w[1]] = 0
    
    num_of_pos    =  len(pos.keys())
    one_hot       =  np.zeros(num_of_pos)
    count = 0
    for key in pos:
        enc         = np.zeros(num_of_pos).tolist()
        enc[count]  = 1
        pos[key] = enc
        count+=1

    return pos
    
def chunks_to_onehot_dictionary(raw_data_filename):
    sentences = split_to_ner_sentences(read_train_data(raw_data_filename))
    chunks = {}
    for s in sentences:
        for item in s:
            w = item.split(" ")
            chunks[w[2]] = 0
    
    num_of_chunks    =  len(chunks.keys())
    one_hot       =  np.zeros(num_of_chunks)
    count = 0
    for key in chunks:
        enc         = np.zeros(num_of_chunks).tolist()
        enc[count]  = 1
        chunks[key] = enc
        count+=1

    return chunks
        
def get_train_test_data_with_glove(raw_data_filename):
    glove_filename           = "glove.6B/glove.6B.50d.txt"
    glove_tokens, mean_vec   =  glove_embeddings(glove_filename)
    sentences = split_to_ner_sentences(read_train_data(raw_data_filename))
    one_hot_labels = labels_to_onehot_dictionary(raw_data_filename)
    y_labels = []
    x_train  = []
    for sent in sentences:
        labels   =  get_labels(sent,one_hot_labels)
        tokens   =  get_tokens(sent)
        x_labels = tokens_from_glove(tokens, glove_tokens,  mean_vec)
        y_labels.append(labels)
        x_train.append(x_labels)
    return x_train, y_labels
    
def get_train_test_data_with_idxs(raw_data_filename):
    one_hot_labels           =  labels_to_onehot_dictionary(raw_data_filename)
    one_hot_pos              =  pos_to_onehot_dictionary(raw_data_filename)
    one_hot_chunks           =  chunks_to_onehot_dictionary(raw_data_filename)
    content                  =  read_train_data(raw_data_filename)
    tokens                   =  get_tokens(content)

    sentences                =  split_to_ner_sentences(content)
    
    vocab                    =  get_vocabulary_as_dict(tokens)
    
    pref1                    =   get_prexif_as_dict(tokens, 0)
    pref2                    =   get_prexif_as_dict(tokens, 1)
    pref3                    =   get_prexif_as_dict(tokens, 2)

    y_labels = []
    x_train  = []
    x_pos    = []
    x_chunks = []
    x_caps   = []
    x_p1     = []
    x_p2     = []
    x_p3     = []
    
    for sent in sentences:
        toks     =  []
        tags     =  []
        chunks   =  []
        capitals =  []
        tokens   =  get_tokens(sent)
        p1=[]
        p2=[]
        p3=[]
        labels   =  get_labels(sent,one_hot_labels)
        pos      =  get_pos_tags(sent)
        chu      =  get_chunks(sent)
        
        for  t in  tokens:
            toks.append(vocab[t]) #  convert tokens to indices for embeddings
            if t[0].isupper():
                capitals.append([1,0])
            else:
                capitals.append([0,1])
                
            p1.append(pref1[t[0]])
            if len(t) > 1:
                p2.append(pref2[t[1]])
            else:
                p2.append(pref2['NOCHAR'])
                
            if len(t) > 2:
                p3.append(pref3[t[2]])
            else:
                p3.append(pref3['NOCHAR'])
        for  p in pos:
            tags.append(one_hot_pos[p])
        for c in chu:
            chunks.append(one_hot_chunks[c])
            
        y_labels.append(labels)
        x_train.append(toks)
        x_pos.append(tags)
        x_chunks.append(chunks)
        x_caps.append(capitals)
        x_p1.append(p1)
        x_p2.append(p2)
        x_p3.append(p3)
    return x_train, x_pos, x_chunks,  x_caps, x_p1, x_p2, x_p3, y_labels, vocab, len(one_hot_pos['NNP']), len(one_hot_chunks['I-NP']) #the last two just take the first one hot encoding simply to return its dimension to be used to initialise the LSTM later on.

def get_only_test_data_with_idxs(raw_data_filename, test_data_filename):
    one_hot_labels           =  labels_to_onehot_dictionary(raw_data_filename)
    one_hot_pos              =  pos_to_onehot_dictionary(raw_data_filename)
    one_hot_chunks           =  chunks_to_onehot_dictionary(raw_data_filename)
    content                  =  read_train_data(raw_data_filename)
    tokens                   =  get_tokens(content)
    
    pref1                    =   get_prexif_as_dict(tokens, 0)
    pref2                    =   get_prexif_as_dict(tokens, 1)
    pref3                    =   get_prexif_as_dict(tokens, 2)

    
    content                  =  read_train_data(test_data_filename)
    
    sentences                =  split_to_ner_sentences(content)

    vocab                    =  get_vocabulary_as_dict(tokens)

    y_labels = []
    x_train  = []
    x_pos    = []
    x_chunks = []
    x_caps   = []
    x_p1     = []
    x_p2     = []
    x_p3     = []
    keep=True
    for sent in sentences:

        toks     =  []
        tags     =  []
        chunks   =  []
        capitals =  []
        p1=[]
        p2=[]
        p3=[]
        labels   =  get_labels(sent,one_hot_labels)
        tokens   =  get_tokens(sent)
        
        pos      =  get_pos_tags(sent)

        chu      =  get_chunks(sent)

        for  t in  tokens:
            if t in vocab.keys():
                toks.append(vocab[t]) #  convert tokens to indices for embeddings
            else:
                toks.append(vocab['OOV'])
                
                
            if t[0].isupper():
                capitals.append([1,0])
            else:
                capitals.append([0,1])
            if t[0] in pref1.keys():
                p1.append(pref1[t[0]])
            else:
                keep=False
                
            if len(t) > 1:
                if t[1] in pref2.keys():
                    p2.append(pref2[t[1]])
                else:
                    keep=False
            else:
                p2.append(pref2['NOCHAR'])
                
            if len(t) > 2:
                if t[2] in pref3.keys():
                    p3.append(pref3[t[2]])
                else:
                    keep=False
            else:
                p3.append(pref3['NOCHAR'])

        for  p in pos:
            tags.append(one_hot_pos[p])
        for c in chu:
            if c in one_hot_chunks.keys():
                chunks.append(one_hot_chunks[c])
            else:
                keep=False
            
#        print(toks)
#        print(tags)
#        print(chunks)
#        input()
#        print("----------NEW SENTENCE---------")
        if keep == True:
            y_labels.append(labels)
            x_train.append(toks)
            x_pos.append(tags)
            x_chunks.append(chunks)
            x_caps.append(capitals)
            x_p1.append(p1)
            x_p2.append(p2)
            x_p3.append(p3)
        keep=True
    return x_train, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, y_labels, vocab, len(one_hot_pos['NNP']), len(one_hot_chunks['I-NP']) #the last two just take the first one hot encoding simply to return its dimension to be used to initialise the LSTM later on.
        
def main():
    nltk.download('punkt')
    nltk.download('stopwords')

    raw_data_filename = "eng.train"
#    glove_filename    = "glove.6B/glove.6B.50d.txt"
#    content           =  read_train_data(raw_data_filename)
#    sentences         =  split_to_ner_sentences(content)
#    tokens            =  get_tokens(content)
#    vocab             =  get_vocabulary_as_dict(tokens)
#    glove_tokens, mean_vec   =  glove_embeddings(glove_filename)
#
#    g_toks            =  tokens_from_glove(tokens, glove_tokens, mean_vec)
#    print(g_toks)
#
    x_train, y_train = get_train_test_data_with_idxs(raw_data_filename)
    print(x_train,  "<->", y_train)
    
#main()
# . . O O (change sentence)
