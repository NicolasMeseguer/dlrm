import numpy as np

def convertMultiHot(arrayOfIndex, sizeofArray):
	initialArr = np.zeros((sizeofArray), dtype=int)
	for i in arrayOfIndex:
		initialArr[i] = 1
	return initialArr

def algEmbeddingMatrix(EmbeddingTable, sparse_index, sparse_offset):
	# Input params:
	# EmbeddingTable -> Embedding Bag
	# sparse_index -> sparse index corresponding to the actual Embedding Bag
	# sparse_offset -> sparse offset used to determine how many lookups we will do on the embedding table.


	# 1. Create a replica of the EmbeddingTable in a matrix W of size nxm
	#	n -> # of embeddings
	n = EmbeddingTable.num_embeddings
	
	#	m -> dim of the embeddings
	m = EmbeddingTable.embedding_dim
	
	#	type of operation
	mode = EmbeddingTable.mode
	
	W = EmbeddingTable.weight.detach().numpy()
	# print(W);print("\n")
	
	# 2. Create matrixes of the offsets and inputs to iterate over and over
	lS_i = sparse_index.detach().numpy()
	# lS_i = np.array([0,2,0,1,2,3])
	lS_o = sparse_offset.detach().numpy()
	# lS_o = np.array([0,2,5])
	
	lookupsResult = []
	# Main Loop. Will iterate L times, lookups.
	for i in range(len(lS_o)):
		# 1. Retrieve the indices of i-th lookup (L).
		starting_pos = lS_o[i]
		if(i+1 == len(lS_o)):
			index_lookup = lS_i[starting_pos: ]
		else:
			ending_pos = lS_o[i+1]
			index_lookup = lS_i[starting_pos:ending_pos]
		
		# 2. Convert the index lookup to a multi-hot vector
		MHVector = convertMultiHot(index_lookup, n)
		
		# 3. Sparse Dense mult to get the value.
		if(mode == "sum"):
			outputVector = MHVector.dot(W)
		elif(mode == "mean"):
			print("modo mean")
		else:
			print("modo max")
		
		lookupsResult.append(outputVector)
	return lookupsResult