''' 
data_parse.py
Contains tools for parsing High Energy Physics data
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 
import ROOT
from ROOT import TTree
import pandas as pd
# import e

def generate_obj_leaves(objname, columns):
	out = []
	for col in columns:
		out.append(objname + "." + col)
	return out 

def extract_ROOT_data_to_hdf(inputfilepath, outputfilepath , leaves,
							trees=None,
							columns=None,
							entrylabel="Entry",
							verbosity=1):
	'''Extracts values from a .root file and writes them to pandas frame stored as an hdf5 file.
		 Essentially takes root data out of its tree format and puts it in a table.
    # Arguments
        inputfilepath: The path to the .root file held locally or in a distributed system
        outputfilepath: The path to the HDF5 file
        leaves: A list of the names of the leaves that need to be extracted 
        trees: A list of the trees in the .root file that need to be parsed.
        	Reads all trees in the file if not set.
        columns: A list of the column names for the table. Uses the leaf names
        	if not seet
        verbosity: 0:No print output, 1:Some, 2:A lot of the table is printed
    '''
	f = ROOT.TFile.Open(inputfilepath)

	if(verbosity > 0): print("Extracting data from " + inputfilepath)

	if(trees == None):
		trees = [x.GetName() for x in f.GetListOfKeys() if isinstance(x.ReadObj(), TTree)]
	if(verbosity > 0): print("Using trees: " + ', '.join(trees))
	if(verbosity > 0): print("Extracting Leaves: " + ', '.join(leaves))
	
	if(columns == None):
		columns = leaves
	else:
		assert len(leaves) == len(columns), "columns input length mismatch: \
			len(leaves) = %r , len(columns) = %r" % (len(leaves), len(columns))
		if(verbosity > 0): print("Renaming to: " + ', '.join(columns))

	dataDict = {}
	dataDict[entrylabel] = []
	for name in columns:
		dataDict[name] = []

	for tree_name in trees:
		tree = f.Get(tree_name)
		l_leaves = []
		for leaf in leaves:
			
			l_leaf = tree.GetLeaf(leaf)
			if(isinstance(l_leaf,ROOT.TLeafElement) == False):
				raise ValueError("Leaf %r does not exist in tree %r." % (leaf,tree_name))

			l_leaves.append(l_leaf)

		n_entries=tree.GetEntries()
		for entry in range(n_entries):
			#
			tree.GetEntry(entry)
			nValues = l_leaves[0].GetLen()
			dataDict[entrylabel] = dataDict[entrylabel] + [entry]*nValues
			for j, l_leaf in enumerate(l_leaves):
				assert l_leaf.GetLen() == nValues, "%r entries in leaf '%r' does not match  \
					%r entries in leaf '%r'" % (l_leaf.GetLen(),l_leaf.GetName(), nValues, l_leaves[0].GetName())
				for i in range(nValues):
					dataDict[columns[j]].append(l_leaf.GetValue(i))


	dataframe = pd.DataFrame(dataDict, columns=[entrylabel]+columns)
	# if(verbosity > 0):
	if(verbosity > 1):
		with pd.option_context('display.max_rows', 999, 'display.max_columns', 10): 
			print(dataframe)
	elif(verbosity > 0):
		with pd.option_context('display.max_rows', 10 , 'display.max_columns', 10): 
			print(dataframe)

	if(verbosity > 0): print("Exporting To " + outputfilepath)
	dataframe.to_hdf(outputfilepath,'data')