import pandas
from Bio import Phylo
from Bio import SeqIO

records = list(SeqIO.parse("PR.fasta", "fasta"))
tree = Phylo.read("neg/clustalo-I20170227-001855-0311-39918476-es.ph", "newick")
# dataset = pandas.read_csv('training_data.csv').values

counter = 0
for clade in tree.clade.clades:
	# c = [ 0 for i in range(2) ]
	# for node in clade.get_terminals():
		# c[dataset[int(node.name) - 1][1]] = c[dataset[int(node.name) - 1][1]] + 1
	# print c
	SeqIO.write([records[int(node.name) - 1] for node in clade.get_terminals()], "neg/PR_neg_%d.fasta" % (counter), "fasta")
	counter = counter + 1