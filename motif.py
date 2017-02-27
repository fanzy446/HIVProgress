from Bio import SeqIO
from pssm import pssm
for i in range(3):
	records = list(SeqIO.parse("PR_neg_%d.fasta" % (i), "fasta"))
	print pssm([ record.seq for record in records ])