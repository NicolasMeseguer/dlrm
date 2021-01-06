import torch
import torch.nn as nn
import numpy as np
import functionEmbeddings as fe

ln = [4, 3, 2]	# embeddings length
m = 2 			# matrix sparsity
m_den = 4		# m den

print("\n1. Create embeddings + Initialize weights\n")
emb_l = nn.ModuleList()
for i in range(0,3):
	n = ln[i]
	EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
	print(EE)
	
	W = np.random.uniform(low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)).astype(np.float32)
	print(W)
	EE.weight.data = torch.tensor(W, requires_grad=True)
	
	emb_l.append(EE)
	print("\n")
# Finished creating


# EE =


#	m
#	^
#	|
#	|	embeddingBag 0			embeddingBag 1		embeddingBag 2
#	    __ __ __ __				 __ __ __ 				 __ __ 
#	1  |__|__|__|__|		1   |__|__|__|			1  	|__|__|	
#	0  |__|__|__|__|		0   |__|__|__|			0   |__|__|
#	    0  1  2  3				 0  1  2				 0  1 		--> n


print("\n\n2. Create random data\n")

data_size = 1;mini_batch_size=1;num_batches=0;index=0

num_indices_per_lookup = 10

# For each embedding, we create n lookups.
# Para cada embedding vamos a crear n vectores de indices para buscar en la embedding table.

# Compute batch size = Always 1, due to the batch size args
n = fe.compute_batch_size(data_size, mini_batch_size, num_batches, index)

(denseFeature, offsets, indices) = fe.generate_uniform_input_batch(m_den, ln, n, num_indices_per_lookup)

#
#
#		{indices1},		{indices2},		{indices3}
#

print("\n3. Apply Embeddings\n")

ly = []

contadorEmbedding = 1
for k in range(len(offsets)):
	print("Embedding Table ", contadorEmbedding)
	sparse_index = indices[k]
	sparse_offset = offsets[k]
	
	E = emb_l[k]
	
	V = E(sparse_index, sparse_offset)
	
	print(V);print("\n")
	ly.append(V)
	contadorEmbedding+=1
