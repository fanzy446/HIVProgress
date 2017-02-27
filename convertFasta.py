import pandas
import math

filePos = open("PR_pos.fasta", "w+")
fileNeg = open("PR_neg.fasta", "w+")
dataset = pandas.read_csv('training_data.csv')

for record in dataset.values:
	if record[2]:
		if record[1] == 1:
			filePos.write(">%d\n" % (record[0]))
			filePos.write(record[2] + "\n\n")
		else:
			fileNeg.write(">%d\n" % (record[0]))
			fileNeg.write(record[2] + "\n\n")

file.close()