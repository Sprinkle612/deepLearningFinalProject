# from transformers import RobertaConfig, RobertaModel,RobertaTokenizer
import pickle
import torch
import spacy
import time

with open ("txts.pkl",'rb') as f:
    txts = pickle.load(f)
print('total sentences ', len(txts))
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')

def encode_roberta(s):
    input_ids = torch.tensor(tokenizer.encode(s)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0] 
    return torch.mean(last_hidden_states[0],axis=0)

# spacy encoding 
nlp = spacy.load('en_core_web_md')
encoded_txts = []
cnt = 0
print('start embedding!')
start = time.time()
for s in txts:
    doc = nlp(s)
    encoded_txts.append(doc.vector)
    cnt += 1
    if cnt % 1000 == 0:
        print('time per 1000 sentences: ', (time.time() - start)/60, ' mins')
        start = time.time()
        with open('encoded_txts.pkl','wb') as f:
            pickle.dump(encoded_txts,f)
            print('%d is saved'%cnt) 

print('encoded sentence length:', len(encoded_txts))
print(len(encoded_txts[0]))

with open('encoded_txts.pkl','wb') as f:
    pickle.dump(encoded_txts,f)

print('pickle saved')

with open('encoded_txts.pkl','rb') as f:
    encoded_txts = pickle.load(f)
print('open pickle,',len(encoded_txts))
