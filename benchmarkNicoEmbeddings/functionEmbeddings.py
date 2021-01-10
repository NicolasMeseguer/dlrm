import numpy as np
from numpy import random as ra
import torch

def compute_batch_size(data_size, mini_batch_size, num_batches, index):
    nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
    if num_batches != 0:
        nbatches = num_batches
        data_size = nbatches * mini_batch_size
    n =  min(mini_batch_size, data_size - (index * mini_batch_size))
    return n

def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
):
    print("Preparing input data: \n")

    print("Dense feature:")
    denseFeature = torch.tensor(ra.rand(n, m_den).astype(np.float32))
    print(denseFeature);print("\n")

    print("Number of lookups per embedding table: ", n)
    print("\n")
    embeddingCounter = 1

    lS_emb_offsets = []
    lS_emb_indices = []

    for size in ln_emb:
		
        print("Embedding num:", str(embeddingCounter))
		
        offsets_current_embedding = []
        indices_current_embedding = []
        offset = 0
		
		# number of sparse indices to be used per embedding
        r = ra.random(1)
        sparse_group_size = np.int64(np.round(max([1.0], r * min(size, num_indices_per_lookup))))
        print("# of sparse indices to be used in the embedding: ", sparse_group_size)
		
        # sparse indices to be used per embedding
        r = ra.random(sparse_group_size)
        sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
        print("Sparse indices to be used in the embedding: ", sparse_group)
		
		# reset sparse_group_size in case some index duplicates were removed
        sparse_group_size = np.int64(sparse_group.size)
		
        offsets_current_embedding = [offset]
        indices_current_embedding = sparse_group.tolist()
		
        print("Offsets ", offsets_current_embedding)
        print("Indices ", indices_current_embedding)
        print("\n\n")
		
        lS_emb_offsets.append(torch.tensor(offsets_current_embedding))
        lS_emb_indices.append(torch.tensor(indices_current_embedding))
        embeddingCounter+=1
    return (denseFeature, lS_emb_offsets, lS_emb_indices)