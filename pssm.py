def pssm(sequences):
	snum = len(sequences[0])
	pssm = [{} for i in range(snum)]
	for sequence in sequences:
		for i in range(len(sequence)):
			if sequence[i] not in pssm[i]:
				pssm[i][sequence[i]] = 0
			pssm[i][sequence[i]] = pssm[i][sequence[i]] + 1

	return "".join([ max(p, key=p.get) for p in pssm ])
