# from Bio.Align.Applications import MuscleCommandline
# cline = MuscleCommandline(input="PR.fasta", out="PR.txt")
# print(cline)
# 
import pandas
from Bio import Phylo
from StringIO import StringIO
from Bio import SeqIO

records = list(SeqIO.parse("PR.fasta", "fasta"))
# tree = Phylo.read(StringIO('(A,(B,C),(D,E));'), 'newick')
tree = Phylo.read("pos/clustalo-I20170227-004309-0305-44032023-pg.ph", "newick")
dataset = pandas.read_csv('training_data.csv').values

counter = 0
for clade in tree.clade.clades:
	# c = [ 0 for i in range(2) ]
	# for node in clade.get_terminals():
	# 	c[dataset[int(node.name) - 1][1]] = c[dataset[int(node.name) - 1][1]] + 1
	# print c
	SeqIO.write([records[int(node.name) - 1] for node in clade.get_terminals()], "PR_pos_%d.fasta" % (counter), "fasta")
	counter = counter + 1
	


