from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pssm import pssm
from malign import malign

# multiple alignment on clusters
# for i in range(3):
# 	# malign("neg/PR_neg_%d.fasta" % (i))

# pssm
centroids = []
for i in range(3):
	records = list(SeqIO.parse("neg/PR_neg_%d.clustal" % (i), "clustal"))
	centroids.append(pssm([ record.seq for record in records ]))

SeqIO.write([SeqRecord(Seq(c), id=str(i)) for i, c in enumerate(centroids)], "neg/centroids.fasta", "fasta")

# comparison
malign("neg/centroids.fasta")