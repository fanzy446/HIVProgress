def pssm(sequences):
	snum = len(sequences[0])
	pssm = [{ 'A': 0, 'T': 0, 'C': 0, 'G': 0 } for i in range(len(sequences[0]))]
	for sequence in sequences:
		for i in range(len(sequence)):
			pssm[i][sequence[i]] = pssm[i][sequence[i]] + 1

	return [ max(p, key=p.get) for p in pssm ]
