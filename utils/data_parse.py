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

class DataProcessingProcedure():

	 def __init__(self, func, input_leaves, output_names):
	 	self.func = func
	 	self.input_leaves = input_leaves
	 	self.output_names = output_names
	 

	 def __call__(self, inputs):
	 	# out = 
	 	# print(out)
	 	return self.func(inputs)


def generate_obj_leaves(objname, columns):
	out = []
	out_col = []
	for col in columns:
		if(isinstance(col,str)):
			out.append(objname + "." + col)
			out_col.append(col)
		elif(isinstance(col,DataProcessingProcedure)):
			out.append(col)
			for name in col.output_names:
				out_col.append(name)
		else:
			out.append(col)
			out_col.append(col)
	return out, out_col

def ROOT_to_pandas(inputfilepath,
					leaves,
					trees=None,
					columns=None,
					entrylabel="Entry",
					verbosity=1):
	'''Extracts values from a .root file and writes them to a pandas frame.
		 Essentially takes root data out of its tree format and puts it in a table.
    # Arguments
        inputfilepath: The path to the .root file held locally or in a distributed system
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
	
	writeColumns = False
	if(columns == None):
		columns = []
		writeColumns = True

	columnmap = {}
	output_length = 0
	leaf_names = []
	col_iter = 0
	for leaf in leaves:
		if(isinstance(leaf,DataProcessingProcedure)):
			output_length += len(leaf.output_names)
			leaf_names = leaf_names + leaf.input_leaves
			if(writeColumns):
				columns = columns + leaf.output_names
			if(verbosity > 0):
				print("Procedure at column %r maps %r -> %r" % (col_iter+1, leaf.input_leaves, leaf.output_names))
			col_iter += len(leaf.output_names)

		else:
			leaf_names.append(leaf)
			if(writeColumns):
				columns.append(leaf)
			try:
				columnmap[leaf] = columns[col_iter]
			except (IndexError):
				raise ValueError("Too few Columns for evaluated output: \n Columns:%r" % columns)
							
			col_iter += 1
			output_length += 1

	if(columns == None):
		columns = leaf_names
		

	if(verbosity > 0):
		print("Extracting Leaves: " + ', '.join(leaf_names))
	if(~writeColumns):
		if(verbosity > 0): print("Renaming to: " + ', '.join(columns))

	assert output_length == len(columns), "columns input length mismatch: \
			len(leaves) = %r , len(columns) = %r" % (output_length, len(columns))


	dataDict = {}
	dataDict[entrylabel] = []
	for name in columns:
		dataDict[name] = []



	for tree_name in trees:
		tree = f.Get(tree_name)
		def get_leaf(leaf):
			try:
				return tree.GetLeaf(leaf)
			except ReferenceError:
				raise ValueError("Tree %r has no leaf %r." % (tree.GetName(), leaf))
		l_leaves = []
		procedures = []
		for leaf in leaves:
			if(isinstance(leaf,DataProcessingProcedure)):
				procedures.append(leaf)
			else:
				l_leaf = get_leaf(leaf)
				if(isinstance(l_leaf,ROOT.TLeafElement) == False):
					raise ValueError("Leaf %r does not exist in tree %r." % (leaf,tree_name))
				l_leaves.append(l_leaf)

		n_entries=tree.GetEntries()
		for entry in range(n_entries):
			tree.GetEntry(entry)
			if(len(l_leaves) > 0 ):
				nValues = l_leaves[0].GetLen()
			elif(len(procedures) > 0):
				nValues = get_leaf(procedures[0].input_leaves[0]).GetLen()
			else:
				nValues = 0

			dataDict[entrylabel] = dataDict[entrylabel] + [entry]*nValues
			for j, l_leaf in enumerate(l_leaves):
				assert l_leaf.GetLen() == nValues, "%r entries in leaf '%r' does not match  \
					%r entries in leaf '%r'" % (l_leaf.GetLen(),l_leaf.GetName(), nValues, l_leaves[0].GetName())
				for i in range(nValues):
					dataDict[columnmap[l_leaf.GetName()]].append(l_leaf.GetValue(i))
			for j, proc in enumerate(procedures):
				for i in range(nValues):
					inputs = []
					for k, leaf in enumerate(proc.input_leaves):
						inputs.append(get_leaf(leaf).GetValue(i))
					out = proc(inputs)
					for k, name in enumerate(proc.output_names):
						dataDict[name].append(out[k])

	
	dataframe = pd.DataFrame(dataDict, columns=[entrylabel]+columns)
	if(verbosity > 1):
		with pd.option_context('display.max_rows', 999, 'display.max_columns', 10): 
			print(dataframe)
	elif(verbosity > 0):
		with pd.option_context('display.max_rows', 10 , 'display.max_columns', 10): 
			print(dataframe)
	return dataframe
	# if(verbosity > 0): print("Exporting To " + outputfilepath)
	# dataframe.to_hdf(outputfilepath,'data')



