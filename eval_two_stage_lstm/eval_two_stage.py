from  utils_bio import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import math
import random
import joblib
import lstm_entities

class Net(nn.Module):
    def __init__(self, embedding_size, pos_tags, chunks, caps_dim, char_emb_size,  device):
        super(Net, self).__init__()
        self.device = device
        
        self.embedding = nn.Embedding(embedding_size, 50)
        self.fc_pos    = nn.Linear(pos_tags, 50)
        self.fc_chunks    = nn.Linear(chunks, 50)
        self.fc_caps    = nn.Linear(caps_dim, 20)
        self.fc_pos_sig = torch.sigmoid
        
        
        
        self.p1  = nn.Embedding(char_emb_size, 20)
        self.p2  = nn.Embedding(char_emb_size, 20)
        self.p3  = nn.Embedding(char_emb_size, 20)

        
        self.lstm = nn.LSTM(230, 50, batch_first=True, dropout=0.2, num_layers=4, bidirectional=False)

        
        self.fc = nn.Linear(50, 3) #3 for BIO
    
    def forward(self, s, tags, chunks, caps, p1,p2,p3):

        
        s = self.embedding(s)
        t = self.fc_pos(tags)
        c = self.fc_chunks(chunks)
        t = self.fc_pos_sig(t)
        caps = self.fc_caps(caps)
        p1 = self.p1(p1)
        p2 = self.p2(p2)
        p3 = self.p3(p3)
 
        s  =  torch.cat((s,t,c,caps, p1,p2,p3),2)


        s, _ = self.lstm(s)

        s = self.fc(s)

        soft = F.log_softmax(s, dim=2)

        return  soft
        

"""
This method is  responsible  for splitting the sentences (:data), pos_tags, labels
into batches. The idea  is that data and pos_tags and chunks will act as inputs to the
LSTM in order to predict named entity tags.
"""
def get_batch(data, pos_tags, chunks, caps, p1,p2,p3, labels, pos_dim, chunks_dim):
        batch_max_len = max([len(s) for s in data]) #find max sentence length
        max_idx =  0 #find maximum tokens, in order to later initialise embedding
        #recall that max_idx works because  the tokens are passed as dictionaries with indices from utils.py
        max_label = 0
        maxp1=0
        maxp2=0
        maxp3=0
        for d in p1:
            for w in d:
                if w > maxp1:
                    maxp1  = w
        for d in p2:
            for w in d:
                if w > maxp2:
                    maxp2  = w
        for d in p3:
            for w in d:
                if w > maxp3:
                    maxp3  = w

        charmax = maxp1
        if  maxp2 > charmax:
            charmax = maxp2
        if maxp3 > charmax:
            charmax =  maxp3
        for d in data:
            for w in d:
                if w > max_idx:
                    max_idx  = w
        for l in labels:
            for i  in l:
                if i > max_label:
                    max_label = i
#        print(max_label)
#        input('waits')
#        print("max_idx", max_idx)
        num_of_sent   = len(data) #number of sentences
        #NOTE: max_ix+2 for OOV when it evaluates on test set
        batch_data    = (max_idx+4)*np.ones((num_of_sent, batch_max_len)) #create a whole batch. Initialise everything to max_idx. max_idx basically corresponds to the tokens we do not want and 'PAD', i.e. added only for torch tensors to work because they can't have varying inputs.
        
        batch_pos     = 0 * np.ones((num_of_sent, batch_max_len, pos_dim)) # Initialise tags batch as one-hot-encoding.
        batch_chunks    = 0 * np.ones((num_of_sent, batch_max_len, chunks_dim))
        batch_caps      = 0 * np.ones((num_of_sent, batch_max_len, 2)) # capilatization is always 2 dim  ([1,0] capital, [0,1] no capital)
        batch_p1      = (maxp1+1) * np.ones((num_of_sent, batch_max_len)) # Note:By design the maxp is the NOCHAR from the utils.py file so no need to increment
        batch_p2      = (maxp2+1) * np.ones((num_of_sent, batch_max_len))
        batch_p3      = (maxp3+1) * np.ones((num_of_sent, batch_max_len))

        
        batch_labels = -1*np.ones((num_of_sent, batch_max_len))# batch labels. Init all to -1, where -1 corresponds to the PADded words. It will be updated in the next loop to correspond to the actual ones.
        for j in range(num_of_sent):
            cur_len = len(data[j]) # Find current sentence length
            batch_data[j][:cur_len] = data[j] # Up to the actual sentence length assign the embeddings indices to the actual tokens, compared to max_idx which was originally initialized to. Note here that the words that do not correspond to the actual sentence remain max_idx. This will be useful later on, when obtaining the loss.

            batch_labels[j][:cur_len] = labels[j] # Do the same for labels.
            batch_pos[j][:cur_len] = pos_tags[j] # Do the  same for tags. Note that tags are in one-hot-encoded versions
            batch_chunks[j][:cur_len] = chunks[j] # Do the  same for tags. Note that tags are in one-hot-encoded versions
            batch_caps[j][:cur_len]   = caps[j]
            batch_p1[j][:cur_len]=p1[j]
            batch_p2[j][:cur_len]=p2[j]
            batch_p3[j][:cur_len]=p3[j]
            
        return batch_data, batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_labels, max_idx+7, charmax+2 #return.max_idx+1 is simply in order to initialize the embedding of the network later on.

"""
Takes  as input  data, pos_tags, chunks, labels and number of batches to produce.
Initially it converts the data to a pytorch batch and the splits them.
"""
def split_to_batches(data, pos_tags, chunks, caps, p1,p2,p3, labels, pos_dim,  chunks_dim, b_num):
    batch_x, batch_pos, batch_chunk, batch_capitals, batch_p1, batch_p2, batch_p3, batch_y, _, _ = get_batch(data, pos_tags, chunks, caps, p1,p2,p3, labels,  pos_dim, chunks_dim) #get batched data in torch tensors.

#    print(batch_pos[1])
#    print(batch_chunk[1])
#    print(batch_y[1])
#
    num_of_sent = len(data)
    batches = int(num_of_sent/b_num)  #determine  batches to split
    batched_x = []
    batched_y = []
    batched_pos=[]
    batched_chunks = []
    batched_caps =[]
    batched_p1 = []
    batched_p2 = []
    batched_p3 = []


    for i in range(batches): #split in batches
        bx = batch_x[i*b_num:(i*b_num + b_num)]
        by = batch_y[i*b_num:(i*b_num + b_num)]
        bp = batch_pos[i*b_num:(i*b_num + b_num)]
        bc = batch_chunk[i*b_num:(i*b_num + b_num)]
        bp1 = batch_p1[i*b_num:(i*b_num + b_num)]
        bp2 = batch_p2[i*b_num:(i*b_num + b_num)]
        bp3 = batch_p3[i*b_num:(i*b_num + b_num)]


        bcaps=batch_capitals[i*b_num:(i*b_num + b_num)]
        
        batched_x.append(bx)
        batched_y.append(by)
        batched_pos.append(bp)
        batched_chunks.append(bc)
        batched_caps.append(bcaps)
        batched_p1.append(bp1)
        batched_p2.append(bp2)
        batched_p3.append(bp3)

    return torch.LongTensor(batched_x), torch.LongTensor(batched_y),  torch.FloatTensor(batched_pos), torch.FloatTensor(batched_chunks), torch.FloatTensor(batched_caps), torch.LongTensor(batched_p1),torch.LongTensor(batched_p2),torch.LongTensor(batched_p3)


def loss_fn(outputs, labels, device):
    count = 0
    #Make extra tokens,i.e., max_idx (from the get_batches method) indices to be zeros.This will later bbe used to ensure they are NOT taken into account when calculating the loss.
    mask = (labels >= 0).float()
    mask = mask.to(device)
    #the number of tokens is the sum of elements in mask
    num_tokens  = int(torch.sum(mask).item()) # just get num of tokens to devide later on.
    exp_log_lik = 0
    out = torch.zeros((outputs.shape[0],outputs.shape[1])).to(device)
    for idx, o in enumerate(outputs):
        for i, tok in enumerate(o):
            if random.random() < 0.90 and labels[idx, i] == 0:
                out[idx, i]= 0
            else:
                out[idx, i]= tok[labels[idx,i]]

            count=count+1
    out=out*mask # Make sure that all  the PAD words  are 0.
#    print(torch.sum(out)/num_tokens)
#    input('wait')
    
    return -torch.sum(out)/num_tokens #expected log likelihood (Recall that the LSTM returns log_softmax.
    
def loss_fn2(outputs, labels, device):
   count = 0
   #Make extra tokens,i.e., max_idx (from the get_batches method) indices to be zeros.This will later bbe used to ensure they are NOT taken into account when calculating the loss.
   mask = (labels >= 0).float()
   mask = mask.to(device)
   #the number of tokens is the sum of elements in mask
   num_tokens  = int(torch.sum(mask).item()) # just get num of tokens to devide later on.
   exp_log_lik = 0
   print(outputs)
   print(labels)
   
   for i, lab in enumerate(labels):
       for l in lab:
           if l !=1:
                exp_log_lik+=outputs[0][i][1]
               
                
#   out=out*mask # Make sure that all  the PAD words  are 0.
   print(exp_log_lik/num_tokens)
   
   return -exp_log_lik/num_tokens #expected log likelihood (Recall that the LSTM returns log_softmax.
   

 
def evaluate(model_bio, model_entities, raw_data_filename, test_data_file):
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        x_test, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, y_test, _, pos_dim, chunks_dim = get_only_test_data_with_idxs(raw_data_filename, test_data_file)
        batch_x,batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_y, max_dim,_ = get_batch(x_test, x_pos, x_chunks, x_caps, x_p1,x_p2, x_p3, y_test, pos_dim, chunks_dim)
        batch_x         = torch.LongTensor(batch_x)
        batch_pos       = torch.FloatTensor(batch_pos)
        batch_chunks    = torch.FloatTensor(batch_chunks)
        batch_caps      = torch.FloatTensor(batch_caps)
        batch_y         = torch.LongTensor(batch_y)
        batch_p1        = torch.LongTensor(batch_p1)
        batch_p2        = torch.LongTensor(batch_p2)
        batch_p3        = torch.LongTensor(batch_p3)
        
        
        batch_x         = batch_x.to(device)
        batch_pos       = batch_pos.to(device)
        batch_chunks    = batch_chunks.to(device)
        batch_caps      = batch_caps.to(device)
        batch_p1        = batch_p1.to(device)
        batch_p2        = batch_p2.to(device)
        batch_p3        = batch_p3.to(device)
        
        split_test          = 0
        
        batch_x             = batch_x[split_test:]
        batch_pos           = batch_pos[split_test:]
        batch_chunks        = batch_chunks[split_test:]
        batch_caps          = batch_caps[split_test:]
        batch_y             = batch_y[split_test:]
        batch_p1            = batch_p1[split_test:]
        batch_p2            = batch_p2[split_test:]
        batch_p3            = batch_p3[split_test:]
        print("evaluation batch_x.shape",batch_x.shape)
        # The  reason we don't split in batches here is because I treat the input from the test set as one batch.
        t = model_bio(batch_x, batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3)
        lab_enc = labels_to_onehot_dictionary(test_data_file)
        lab_num_to_dict = {}
        z=0
        for key in lab_enc.keys():
            k = "{}".format(z)
            lab_num_to_dict[k] = key
            z+=1
        print(lab_num_to_dict)
        results=[]
        bio_predictions = []
        max_len  =  0
#        for qwerty in t:
#            temp = []
#            c =  0
#            for res in qwerty:
#                results.append(torch.argmax(res).item())
#                if  res.argmax(0).item() == 0:
#                    temp.append([0,0,1])
#                elif res.argmax(0).item()  == 1:
#                    temp.append([0,1,0])
#                else:
#                    temp.append([1,0,0])
#                c+=1
#            if c > max_len:
#                max_len = c
#
#
        for qwerty in t:
            temp = []
            for res in qwerty:
                results.append(torch.argmax(res).item())
                temp.append(res.argmax(0).item())

        k=0
        for i, b in enumerate(batch_y):
            temp = []
            for idx, l in enumerate(b):
                if l!=-1:
                    if  results[k] == 0:
                        temp.append([0,0,1])
                    elif results[k]  == 1:
                        temp.append([0,1,0])
                    else:
                        temp.append([1,0,0])
                k+=1
            
            # {'O': 0, 'I': 1, 'B': 2}

            bio_predictions.append(temp)
            
        x_test, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, x_bio, y_test, _, pos_dim, chunks_dim = lstm_entities.get_only_test_data_with_idxs(raw_data_filename, test_data_file)
#        print("max length", max_len)
        x_= bio_predictions # replace with the ones from the first LSTM
        
        batch_x,batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_bio, batch_y, max_dim,_ = lstm_entities.get_batch(x_test, x_pos, x_chunks, x_caps, x_p1,x_p2, x_p3,x_bio, y_test, pos_dim, chunks_dim)
        batch_x         = torch.LongTensor(batch_x)
        batch_pos       = torch.FloatTensor(batch_pos)
        batch_chunks    = torch.FloatTensor(batch_chunks)
        batch_caps      = torch.FloatTensor(batch_caps)
        batch_y         = torch.LongTensor(batch_y)
        batch_p1        = torch.LongTensor(batch_p1)
        batch_p2        = torch.LongTensor(batch_p2)
        batch_p3        = torch.LongTensor(batch_p3)
        batch_bio       = torch.FloatTensor(batch_bio)
        
        
        batch_x         = batch_x.to(device)
        batch_pos       = batch_pos.to(device)
        batch_chunks    = batch_chunks.to(device)
        batch_caps      = batch_caps.to(device)
        batch_p1        = batch_p1.to(device)
        batch_p2        = batch_p2.to(device)
        batch_p3        = batch_p3.to(device)
        batch_bio       = batch_bio.to(device)
        split_test          = 0
        
        batch_x             = batch_x[split_test:]
        batch_pos           = batch_pos[split_test:]
        batch_chunks        = batch_chunks[split_test:]
        batch_caps          = batch_caps[split_test:]
        batch_y             = batch_y[split_test:]
        batch_p1            = batch_p1[split_test:]
        batch_p2            = batch_p2[split_test:]
        batch_p3            = batch_p3[split_test:]
        batch_bio           = batch_bio[split_test:]
        print("evaluation batch_x.shape",batch_x.shape)
        # The  reason we don't split in batches here is because I treat the input from the test set as one batch.
        t = model_entities(batch_x, batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_bio)
        lab_enc = labels_to_onehot_dictionary(test_data_file)
        lab_num_to_dict = {}
        z=0
        for key in lab_enc.keys():
            k = "{}".format(z)
            lab_num_to_dict[k] = key
            z+=1
        print(lab_num_to_dict)
        results=[]
        rando=[]
        for qwerty in t:
            temp = []
            for res in qwerty:
                results.append(torch.argmax(res).item())
                temp.append(res.argmax(0).item())

            rando.append(temp)
                    # {'O': 0, 'I': 1, 'B': 2}
        print(batch_y)
        print(rando)
        per=[]
        org=[]
        misc=[]
        loc=[]
        other=[]
        m=0
        c=0
        k=0
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != l and l == 0:
                        fp+=1
                    elif results[k] == l and l == 0:
                        tn+=1
                    elif results[k] !=l and results[k] == 0:
                        fn+=1
                    elif results[k] ==l and l!=0:
                        tp+=1
#                    '0': 'O', '1': 'I-ORG', '2': 'I-LOC', '3': 'I-MISC', '4': 'I-PER', '5': 'B-MISC', '6': 'B-ORG', '7': 'B-LOC'}
                    
                if l!=-1 and l!=0:
                    if results[k] == l:
                        c+=1
                    m+=1
                k+=1
                
        k=0
        tporg=0; fporg=0; fnorg=0;
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != 0  and l==0:
                        fnorg+=1
                    elif results[k] ==0 and l!=0:
                        fporg+=1
                    elif results[k] == 0 and l == 0:
                        tporg+=1
                k+=1
#        {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}

        if tporg > 0:
            p = (tporg/(tporg+fporg)); r = (tporg/(tporg+fnorg))
            print("O label: Precision: ", p, " Recall: ", r, " F1: ", (2*p*r)/(p+r))
        other.append([p,r,(2*p*r)/(p+r)])
        k=0
        tporg=0; fporg=0; fnorg=0;
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != 1  and l==1:
                        fnorg+=1
                    elif results[k] ==1 and l!=1:
                        fporg+=1
                    elif results[k] == 1 and l == 1:
                        tporg+=1
                k+=1
#        {'O': 0, 'ORG': 1, 'MISC': 2, 'PER': 3, 'LOC': 4}

        if tporg > 0:
            p = (tporg/(tporg+fporg)); r = (tporg/(tporg+fnorg))
            print("ORG label: Precision: ", p, " Recall: ", r, " F1: ", (2*p*r)/(p+r))
        org.append([p,r,(2*p*r)/(p+r)])

        k=0
        tporg=0; fporg=0; fnorg=0;
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != 2  and l==2:
                        fnorg+=1
                    elif results[k] ==2 and l!=2:
                        fporg+=1
                    elif results[k] == 2 and l == 2:
                        tporg+=1
                k+=1
#        {'O': 0, 'ORG': 1, 'LOC': 2, 'MISC': 3, 'PER': 4}

        if tporg > 0:
            p = (tporg/(tporg+fporg)); r = (tporg/(tporg+fnorg))
            print("LOC label: Precision: ", p, " Recall: ", r, " F1: ", (2*p*r)/(p+r))
        loc.append([p,r,(2*p*r)/(p+r)])

        k=0
        tporg=0; fporg=0; fnorg=0;
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != 3  and l==3:
                        fnorg+=1
                    elif results[k] ==3 and l!=3:
                        fporg+=1
                    elif results[k] == 3 and l == 3:
                        tporg+=1
                k+=1
#        {'O': 0, 'ORG': 1, 'LOC': 2, 'MISC': 3, 'PER': 4}

        if tporg > 0:
            p = (tporg/(tporg+fporg)); r = (tporg/(tporg+fnorg))
            print("MISC label: Precision: ", p, " Recall: ", r, " F1: ", (2*p*r)/(p+r))
        misc.append([p,r,(2*p*r)/(p+r)])

        k=0
        tporg=0; fporg=0; fnorg=0;
        for i, b in enumerate(batch_y):
            for idx, l in enumerate(b):
                if l!=-1:
                    if results[k] != 4  and l==4:
                        fnorg+=1
                    elif results[k] ==4 and l!=4:
                        fporg+=1
                    elif results[k] == 4 and l == 4:
                        tporg+=1
                k+=1
#        {'O': 0, 'ORG': 1, 'LOC': 2, 'MISC': 3, 'PER': 4}

        if tporg > 0:
            p = (tporg/(tporg+fporg)); r = (tporg/(tporg+fnorg))
            print("PER label: Precision: ", p, " Recall: ", r, " F1: ", (2*p*r)/(p+r))
        per.append([p,r,(2*p*r)/(p+r)])

           
                
        print("Naive Accuracy: {}".format(c/m))
#        print("batchy", batch_y)
    if tp > 0:
        return org, misc, per, loc, other
    return 0, 0

def get_bio_model(raw_data_filename):
    batches=512

    x_train, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, y_train, _, pos_dim, chunks_dim = get_train_test_data_with_idxs(raw_data_filename)

    batched_x, batched_y, batched_pos, batched_chunks, batched_capitalisations, batched_p1, batched_p2, batched_p3 = split_to_batches(x_train, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, y_train, pos_dim, chunks_dim ,batches)


    batch_x,batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_y, max_dim, char_emd_dim = get_batch(x_train, x_pos, x_chunks, x_caps,x_p1,x_p2,x_p3, y_train, pos_dim, chunks_dim)
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    print("Batch shape:", batched_x.shape)
    print("Device:", device)
    lstm_net = Net(max_dim, pos_dim,chunks_dim, 2, char_emd_dim, device).to(device) #2 is  for the capitalisation one hot encoding dimension
    return lstm_net
    
def get_entities_model(raw_data_filename):
    batches=512

    x_train, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, x_bio, y_train, _, pos_dim, chunks_dim = lstm_entities.get_train_test_data_with_idxs(raw_data_filename)

    batched_x, batched_y, batched_pos, batched_chunks, batched_capitalisations, batched_p1, batched_p2, batched_p3, batched_bio = lstm_entities.split_to_batches(x_train, x_pos, x_chunks, x_caps, x_p1, x_p2, x_p3, x_bio, y_train, pos_dim, chunks_dim ,batches)


    batch_x,batch_pos, batch_chunks, batch_caps, batch_p1, batch_p2, batch_p3, batch_bio, batch_y, max_dim, char_emd_dim = lstm_entities.get_batch(x_train, x_pos, x_chunks, x_caps,x_p1,x_p2,x_p3, x_bio, y_train, pos_dim, chunks_dim)
    
    
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    lstm_net = lstm_entities.Net(max_dim, pos_dim,chunks_dim, 2, char_emd_dim, 3, device).to(device) #2 is  for the capitalisation one hot encoding dimension
    return lstm_net
import statistics
 
def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    batches=512
    raw_data_filename = "eng.train"
    eval_file="eng.testb"
    bio_model  = get_bio_model(raw_data_filename)
    entities_model =  get_entities_model(raw_data_filename)
    forg=[]
    fmisc=[]
    fper=[]
    floc=[]
    fother=[]
    for i in [2, 7, 11]:
        bio_model.load_state_dict(torch.load('lstm1bio.pt', map_location=torch.device('cpu')))
        entities_model.load_state_dict(torch.load("lstm_entities_seed_{}.pt".format(i), map_location=torch.device('cpu')))

        org, misc, per, loc, other = evaluate(bio_model, entities_model, raw_data_filename, eval_file)
        
        forg.append(org[0])
        fmisc.append(misc[0])
        fper.append(per[0])
        floc.append(loc[0])
        fother.append(other[0])
    
#        print("Precision: ", lp,"Recall: ", lr)
#
#     print("ORG p: ",statistics.mean([forg[0][0],forg[1][0],forg[2][0]]) ,"st: ", statistics.stdev([forg[0][0],forg[1][0],forg[2][0]])))
#     print("ORG r: ",statistics.mean([forg[0][1],forg[1][1],forg[2][1]]) ,"st: ", statistics.stdev([forg[0][1],forg[1][1],forg[2][1]]))
#     print("ORG f: ",statistics.mean([forg[0][2],forg[1][2],forg[2][2]]) ,"st: ", statistics.stdev([forg[0][2],forg[1][2],forg[2][2]]))
#
    print(forg)
    print("ORG p", statistics.mean([forg[0][0],forg[1][0],forg[2][0]]), statistics.stdev([forg[0][0],forg[1][0],forg[2][0]]))
    print("ORG r", statistics.mean([forg[0][1],forg[1][1],forg[2][1]]), statistics.stdev([forg[0][1],forg[1][1],forg[2][1]]))
    print("ORG f", statistics.mean([forg[0][2],forg[1][2],forg[2][2]]), statistics.stdev([forg[0][2],forg[1][2],forg[2][2]]))
    
    print("MISC p", statistics.mean([fmisc[0][0],fmisc[1][0],fmisc[2][0]]), statistics.stdev([fmisc[0][0],fmisc[1][0],fmisc[2][0]]))
    print("MISC r", statistics.mean([fmisc[0][1],fmisc[1][1],fmisc[2][1]]), statistics.stdev([fmisc[0][1],fmisc[1][1],fmisc[2][1]]))
    print("MISC f", statistics.mean([fmisc[0][2],fmisc[1][2],fmisc[2][2]]), statistics.stdev([fmisc[0][2],fmisc[1][2],fmisc[2][2]]))

    print("LOC p", statistics.mean([floc[0][0],floc[1][0],floc[2][0]]), statistics.stdev([floc[0][0],floc[1][0],floc[2][0]]))
    print("LOC r", statistics.mean([floc[0][1],floc[1][1],floc[2][1]]), statistics.stdev([floc[0][1],floc[1][1],floc[2][1]]))
    print("LOC f", statistics.mean([floc[0][2],floc[1][2],floc[2][2]]), statistics.stdev([floc[0][2],floc[1][2],floc[2][2]]))


    print("PER  p", statistics.mean([fper[0][0],fper[1][0],fper[2][0]]), statistics.stdev([fper[0][0],fper[1][0],fper[2][0]]))
    print("PER  r", statistics.mean([fper[0][1],fper[1][1],fper[2][1]]), statistics.stdev([fper[0][1],fper[1][1],fper[2][1]]))
    print("PER  f", statistics.mean([fper[0][2],fper[1][2],fper[2][2]]), statistics.stdev([fper[0][2],fper[1][2],fper[2][2]]))
    
    print("O p", statistics.mean([fother[0][0],fother[1][0],fother[2][0]]), statistics.stdev([fother[0][0],fother[1][0],fother[2][0]]))
    print("O r", statistics.mean([fother[0][1],fother[1][1],fother[2][1]]), statistics.stdev([fother[0][1],fother[1][1],fother[2][1]]))
    print("O f", statistics.mean([fother[0][2],fother[1][2],fother[2][2]]), statistics.stdev([fother[0][2],fother[1][2],fother[2][2]]))

main()


