from Bio.Align.Applications import ClustalwCommandline

def malign(fileName):
	f = fileName.split(".")[0]
	clustalw_cline = ClustalwCommandline("./clustal", infile=fileName, 
	outfile=f + ".clustal", newtree=f + ".dnd" )
	stdout, stderr = clustalw_cline()

malign("neg/PR_neg.fasta")