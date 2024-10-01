import torch
import esm
import collections
import pandas as pd
from copy import deepcopy
import numpy as np
from tqdm import tqdm


def esm_embeddings(peptide_sequence_list):
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)  
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    return embeddings_results

# load the model
# NOTICE: if the model was not downloaded in your local environment, it will automatically download it.
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

print('Loading checkpoint')
checkpoint = torch.load("./checkpoint-22.pth", map_location='cuda')        
seq_head = "modality_trunks.sequence.seq_encoder."
for keys, value in checkpoint['model'].items():
  if seq_head in keys:
    key = keys.replace(seq_head, '')
    if key in model.state_dict().keys():
      model.state_dict()[key] = deepcopy(value)
    else:
      pass
            
model = model.cuda()
model.eval()  # disables dropout for deterministic results


dataset = pd.read_csv("./generator_train.csv")
sequence_list = dataset['sequence'] 

embeddings_results = pd.DataFrame()
for seq in tqdm(sequence_list):
    format_seq = [seq,seq] # the setting is just following the input format setting in ESM model, [name,sequence]
    tuple_sequence = tuple(format_seq)
    peptide_sequence_list = []
    peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
    # employ ESM model for converting and save the converted data in csv format
    one_seq_embeddings = esm_embeddings(peptide_sequence_list)
    embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])

embeddings_results.to_csv('./esm_embeddings.csv')
