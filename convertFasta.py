import pandas
import math

filePos = open("pos/PR_pos.fasta", "w+")
fileNeg = open("neg/PR_neg.fasta", "w+")
dataset = pandas.read_csv('training_data.csv')

for record in dataset.values:
	if record[1] == 1:
		filePos.write(">%d\n" % (record[0]))
		filePos.write(record[2] + "\n")
	else:
		fileNeg.write(">%d\n" % (record[0]))
		fileNeg.write(record[2] + "\n")

file.close()