''' 
data_parse.py
Contains tools for parsing High Energy Physics data
Author: Danny Weitekamp
e-mail: dannyweitekamp@gmail.com
''' 
import ROOT
from ROOT import TTree
import pandas as pd
import numpy as np
import sys
import time
import gc


class DataProcessingProcedure():
	'''
		An object that can be passed as a leaf in ROOT_to_pandas. Instead of simply grabbing a
		leaf, it takes in data and applies a function to it. The outputs of that function are
		written instead of the leaf to the pandas frame.
	'''
	def __init__(self, func, input_leaves, output_names):
	 	'''
	 		Initialization
	 	# Arguments
			func: a function that maps the inputs (a list or tuple)
				 to the outputs (a list or tuple).
			input_leaves: The fully qualifed names of the leaves whose
				values func will take as inputs
			output_names: The names of the column headers for the outputs
	 	'''
	 	self.func = func
	 	self.input_leaves = input_leaves
	 	self.output_names = output_names
	 	self.input_leaf_objs = []
	 

	def __call__(self, inputs):
	 	return self.func(inputs)

	def __str__(self):
	 	return "%r -> %r" % (self.input_leaves, self.output_names)



def leaves_from_obj(objname, columns):
	'''
		Takes the name of an object and a list of properties (Columns). Expands like So 
		leaves_from_obj("ObjectName", ["Prop1, "Prop2", "Prop3"]) ->
		 ["ObjectName.Prop1, "ObjectName.Prop2", "ObjectName.Prop3"] , ["Prop1, "Prop2", "Prop3"]
		Can also take in DataProcessingProcedures which it will properly expand in the column output
		# Arguments
			objname: The name of the object
			input_leaves: The name of its properties 
		# Returns
			leaves, columns
			leaves: The a list of fully qualified leaf names
			columns: The a list of column names. Expanded if a DataProcessingProcedure was given.
	'''
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

def ROOT_NumValues_to_pandas(inputfilepath,leaves, columns=None, tree_name="Delphes"):
	f = ROOT.TFile.Open(inputfilepath)
	# if(trees == None):
	# 	trees = [x.GetName() for x in f.GetListOfKeys() if isinstance(x.ReadObj(), TTree)]
	#Loop over the tree

	out = {}
	# for tree_name in trees:
	# tree_out = {}
	# out[tree_name] = tree_out
	#if we didn't get any column names, just names just use the fully qualified leaf names
	if(columns == None):
		columns = leaves
	assert len(leaves) == len(columns), "columns input length mismatch: \
			len(leaves) = %r , len(columns) = %r" % (len(leaves), len(columns))

	tree = f.Get(tree_name)
	tree.SetCacheSize(30*1024*1024)
	branches = []
	for leaf_name in leaves: 
		b = tree.GetBranch(leaf_name)
		branches.append(b)
		tree.AddBranchToCache(b)
	tree.StopCacheLearningPhase()

	columnmap = {}
	l_leaves = []
	for i,leaf in enumerate(leaves):
		columnmap[leaf] = columns[i]
		l_leaf = tree.GetLeaf(leaf)
		if(isinstance(l_leaf,ROOT.TLeafElement) == False):
			raise ValueError("Leaf %r does not exist in Tree %r." % (leaf,tree_name))
		l_leaves.append(l_leaf)

	total_values = 0
	n_entries=tree.GetEntries()
	for l_leaf in l_leaves:
		name = l_leaf.GetName()
		out[columnmap[name]] = [None] * n_entries
	for entry in range(n_entries):
		tree.LoadTree(entry)
		for b in branches:
			b.GetEntry(entry)
		tree.GetEntry(entry)
		#Entries have multiple values that we need to extract. Get that number
		
		for l_leaf in l_leaves:
			name = l_leaf.GetName()
			nValues = l_leaf.GetLen()
			out[columnmap[name]][entry] = nValues
	return pd.DataFrame(out)
			



		
		

				

def ROOT_to_pandas(inputfilepath,
					leaves,
					trees=None,
					columns=None,
					addEntry=True,
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
	#Open the root file
	f = ROOT.TFile.Open(inputfilepath)

	if(verbosity > 0):
		last_time = start_time = time.clock()
	if(verbosity > 0): print("Extracting data from " + inputfilepath)

	if(trees == None):
		trees = [x.GetName() for x in f.GetListOfKeys() if isinstance(x.ReadObj(), TTree)]
	if(verbosity > 0): print("Using trees: " + ', '.join(trees))
	
	writeColumns = False
	if(columns == None):
		columns = []
		writeColumns = True

	#Loop over the leaves, and make sure they are in the correct format. Also do some preprocessing.
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
				print("Procedure at column %r maps %r" % (col_iter+1, str(leaf)))
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

	#if we didn't get any column names, just names just use the fully qualified leaf names
	if(columns == None):
		columns = leaf_names

	seen = set()
	seen_add = seen.add
	leaf_names_no_repeat = [x for x in leaf_names if not (x in seen or seen_add(x))]

	if(verbosity > 0):
		print("Extracting Leaves: " + ', '.join(leaf_names_no_repeat))
	if(~writeColumns):
		if(verbosity > 0): print("Renaming to: " + ', '.join(columns))

	assert output_length == len(columns), "columns input length mismatch: \
			len(leaves) = %r , len(columns) = %r" % (output_length, len(columns))


	dataDict = {}
	dataDict[entrylabel] = []
	for name in columns:
		dataDict[name] = []


	#Loop over the tree
	for tree_name in trees:
		tree = f.Get(tree_name)
		tree.SetCacheSize(30*1024*1024)
		branches = []
		for leaf_name in leaf_names_no_repeat: 
			b = tree.GetBranch(leaf_name)
			branches.append(b)
			tree.AddBranchToCache(b)
		tree.StopCacheLearningPhase()

		#Make sure that the leaves all exist in the tree
		l_leaves = []
		procedures = []
		for leaf in leaves:
			if(isinstance(leaf,DataProcessingProcedure)):
				proc = leaf
				procedures.append(proc)
				proc.input_leaf_objs = []
				for l in proc.input_leaves:
					obj = tree.GetLeaf(l)
					if(isinstance(obj,ROOT.TLeafElement) == False):
						raise ValueError("Input Leaf %r in Procedure %r does not exist in Tree %r" % (l,str(proc), tree_name))
					proc.input_leaf_objs.append(obj)

			else:
				l_leaf = tree.GetLeaf(leaf)
				if(isinstance(l_leaf,ROOT.TLeafElement) == False):
					raise ValueError("Leaf %r does not exist in Tree %r." % (leaf,tree_name))
				l_leaves.append(l_leaf)

		#Iterate over all the data and figure out how many total values there are
		total_values = 0
		n_entries=tree.GetEntries()
		for entry in range(n_entries):

			tree.LoadTree(entry)
			for b in branches:
				b.GetEntry(entry)
			tree.GetEntry(entry)
			#Entries have multiple values that we need to extract. Get that number
			if(len(l_leaves) > 0):
				nValues = l_leaves[0].GetLen()
			elif(len(procedures) > 0):
				nValues = (procedures[0].input_leaf_objs[0]).GetLen()
			else:
				nValues = 0
			total_values += nValues

		#Prepopulate every array in the dictionary with values
		for key in dataDict:
			dataDict[key] = dataDict[key] + [None]*total_values


		#Loop over all the entries 
		percent = 0.
		prev_entry = 0
		val_num = 0
		#n_entries=tree.GetEntries()
		for entry in range(n_entries):
			if(verbosity > 0):
				c = time.clock() 
				if(c > last_time + .25):
					percent = float(entry)/float(n_entries)
					sys.stdout.write('\r')
					# the exact output you're looking for:
					sys.stdout.write("[%-20s] %r/%r  %r(Entry/sec)" % ('='*int(20*percent), entry, int(n_entries), 4 * (entry-prev_entry)))
					sys.stdout.flush()
					last_time = c
					prev_entry = entry


			#Point the tree to the next entry <- IMPORTANT this is how we loop
			tree.LoadTree(entry)
			for b in branches:
				b.GetEntry(entry)
			tree.GetEntry(entry)

			#Entries have multiple values that we need to extract. Get that number
			if(len(l_leaves) > 0):
				nValues = l_leaves[0].GetLen()
			elif(len(procedures) > 0):
				nValues = (procedures[0].input_leaf_objs[0]).GetLen()
			else:
				nValues = 0
			
			#Store what entry we are in in the table
			if(addEntry):
				for i in range(nValues):
					# dataDict[entrylabel] = dataDict[entrylabel] + [entry]*nValues
					dataDict[entrylabel][val_num + i] = entry

			#Loop over all the leaves that we just need to copy
			for j, l_leaf in enumerate(l_leaves):
				assert l_leaf.GetLen() == nValues, "%r entries in leaf '%r' does not match  \
					%r entries in leaf '%r'" % (l_leaf.GetLen(),l_leaf.GetName(), nValues, l_leaves[0].GetName())
				#Place all the values in the dictionary
				for i in range(nValues):
					#dataDict[columnmap[l_leaf.GetName()]].append(l_leaf.GetValue(i))
					dataDict[columnmap[l_leaf.GetName()]][val_num + i] = l_leaf.GetValue(i)

			#Loop over all the DataProcessingProcedures
			for j, proc in enumerate(procedures):
				for i in range(nValues):
					inputs = []

					#Loop over all the leaves that we for DataProcessingProcedure inputs
					for k, l_leaf in enumerate(proc.input_leaf_objs):
						inputs.append(l_leaf.GetValue(i))

					#Apply the mapping function
					out = proc(inputs)

					#If we forgot to make the output of our function a list, fix that.
					if isinstance(out, (list, tuple)) == False:
						out = [out]

					#Put our outputs in the dictionary
					for k, name in enumerate(proc.output_names):
						#dataDict[name].append(out[k])
						dataDict[name][val_num + i] = (out[k])

			val_num += nValues
	if(addEntry):
		columns = [entrylabel]+columns
	
	#Make the dataframe from the dictionary
	dataframe = pd.DataFrame(dataDict, columns=columns)
	# f = None
	# t = None
	dataDict = None
	if(verbosity > 1):
		with pd.option_context('display.max_rows', 999, 'display.max_columns', 10): 
			print(dataframe)
	elif(verbosity > 0):
		with pd.option_context('display.max_rows', 10 , 'display.max_columns', 10): 
			print(dataframe)
	if(verbosity > 0):
		print("Reading %r bytes in %d transactions\n" % (f.GetBytesRead(),  f.GetReadCalls()));
		print("Elapse time: %.2f seconds" % float(time.clock()-start_time))
	return dataframe


def getPandasNumValues(filename):
	return ROOT_NumValues_to_pandas(filename,
		["Photon.Phi","Electron.Phi","MuonTight.Phi","MissingET.Phi","EFlowPhoton.Phi","EFlowNeutralHadron.Phi","EFlowTrack.Phi"],
		columns=["Photon","Electron","MuonTight","MissingET","EFlowPhoton","EFlowNeutralHadron","EFlowTrack"]


mass_of_electron = np.float64(0.0005109989461) #eV/c
mass_of_muon = np.float64(0.1056583715)      #eV/c
def four_vec_from_PT(inputs, M):
	'''Helper function that converts (PT, Eta, Phi) -> (E/c, Px, Py, Pz)'''
	PT = inputs[0] 
	Eta = inputs[1]
	Phi = inputs[2]
	#M has units of eV/(c^2)
	if(M == None):
		M = inputs[3]
	momentum_mag = PT * np.cosh(Eta)
   
	E_over_c = np.sqrt(np.square(M) + np.square(momentum_mag))
	px = PT * np.cos(Phi) 
	py = PT * np.sin(Phi) 
	pz = PT * np.sinh(Eta)
	return [E_over_c, px, py, pz]

#DEGENERATE
def getPandasParticles(filename, cull=True):
	#C=1 in natural units so no processing needs to be done
	E_over_c_proc = DataProcessingProcedure(lambda x:x[0], ["Particle.E"], ["E/c"])
	columns= [E_over_c_proc, "Px", "Py", "Pz", "PID", "Charge"]#, "Status"]
	leaves, columns = leaves_from_obj("Particle", columns+["Phi", "Eta"] + dxy_proc + elfow_proc)
	original_frame = ROOT_to_pandas(filename,
								 leaves,
								  columns=columns,
								  verbosity=1)
	if(cull):
		particle_frame = cullNonObservables(original_frame)
	else:
		particle_frame = original_frame
	return particle_frame


def getPandasSpecificObject(filename, name, mass=None, pid=None, chrg_def=-1, charge=None, XT="PT", Dxy_Ehad_Eem=None):
	'''Converts a particular type of ROOT object to table format
		Arguements:
		filaname	  -- Path to the .root filen
		name		  -- The name of the object being converted
		mass=None	 -- The mass of the object being converted, by default reads the mass from the object
		pid=None	  -- The pid of the object, by default reads the pid from the object
		chrg_def=-1   -- The default charge of a particle, (e.g. an electron would be -1) as opposed to its anti-particle
		charge=None   -- The charge, by default reads it from object
		XT="PT"	   -- The transverse measurement. Can be PT, ET, or MET. Used to compute the 4-momentum
		Dxy_Ehad_Eem=None -- A string indicating if the Dxy,Ehad, or Eem properties should be read. They occupy one column since
								they are never mutually non-zero.
		Returns:
		A pandas frame of the format: [E/c, Px, Py, Pz, PID, Charge, PT_ET, Eta, Phi, Dxy_Ehad_Eem]
	'''
	assert Dxy_Ehad_Eem in [None, "Dxy", "Ehad", "Eem"]
	assert XT in ["PT", "ET", "MET"]
	assert mass >= 0.0
	
	if(mass != None):
		four_vec_inputs, dummy = leaves_from_obj(name, [XT, "Eta", "Phi"])
	else:
		four_vec_inputs, dummy = leaves_from_obj(name, [XT, "Eta", "Phi", "Mass"])
		
	four_vec_proc = DataProcessingProcedure(lambda x: four_vec_from_PT(x,mass)
											, four_vec_inputs, ["E/c", "Px","Py","Pz"])
	
	#status_proc = DataProcessingProcedure(lambda x:[1], [], ["Status"])
	
	if(charge == None):
		if(pid != None):
			PID_charge_proc = DataProcessingProcedure(lambda x: [pid*chrg_def*x[0], x[0]]
													  , [name + ".Charge"], ["PID", "Charge"])
			columns=[four_vec_proc, PID_charge_proc]#, status_proc]
		else:
			columns=[four_vec_proc, "PID", "Charge"]#, status_proc]
	else:
		charge_proc = DataProcessingProcedure(lambda x: [charge], [], ["Charge"])
		if(pid != None):
			PID_proc = DataProcessingProcedure(lambda x: [pid], [], ["PID"])
			columns=[four_vec_proc, PID_proc, charge_proc]#,status_proc]
		else:
			columns=[four_vec_proc, "PID", charge_proc]#,status_proc]
		
		
	xt_proc = DataProcessingProcedure(lambda x: [x[0]], [name + "." + XT], ["PT_ET"])
	if(Dxy_Ehad_Eem == None):
		dxy_eflow_proc = DataProcessingProcedure(lambda x: [0], [], ["Dxy_Ehad_Eem"])
	else:
		dxy_eflow_proc = DataProcessingProcedure(lambda x: [x[0]], [name + "."+ Dxy_Ehad_Eem], ["Dxy_Ehad_Eem"])
		
	
	leaves, columns = leaves_from_obj(name, columns+[xt_proc] + ["Phi", "Eta"] + [dxy_eflow_proc])

	#Extract the tables from the root file
	frame = ROOT_to_pandas(filename,
						  leaves,
						  columns=columns,
						  verbosity=1)
	return frame

def getPandasPhotons(filename):
	''' Returns a pandas frame derived from the Photon objects in the given ROOT file'''
	return getPandasSpecificObject(filename, "Photon", mass=0, pid=22, charge=0)

def getPandasElectrons(filename):
	''' Returns a pandas frame derived from the Electron objects in the given ROOT file'''
	return getPandasSpecificObject(filename, "Electron", mass=mass_of_electron, pid=11)

def getPandasTightMuons(filename):
	''' Returns a pandas frame derived from the MuonTight objects in the given ROOT file'''
	return getPandasSpecificObject(filename, "MuonTight", mass=mass_of_muon, pid=13)

def getPandasJets(filename):
	''' Returns a pandas frame derived from the Jet objects in the given ROOT file'''
	return getPandasSpecificObject(filename, "Jet", pid=100, chrg_def=1)

def getPandasMissingET(filename, name="MissingET"):  
	''' Returns a pandas frame derived from the MissingET objects in the given ROOT file'''
	pid = 83
	if(name == "PuppiMissingET"): pid=84
	return getPandasSpecificObject(filename, name, mass=0, pid=pid,charge=0, XT="MET")

def getPandasEFlowParticle(filename, name="EFlowPhoton"):  
	''' Returns a pandas frame derived from the EFlowPhoton objects in the given ROOT file'''
	pid = 90
	Dxy_Ehad_Eem = "Eem"
	if(name == "EFlowNeutralHadron"):
		pid=91
		Dxy_Ehad_Eem = "Ehad"
	return getPandasSpecificObject(filename, name, mass=0, charge=0, pid=pid, XT="ET", Dxy_Ehad_Eem=Dxy_Ehad_Eem)

def getPandasEFlowTrack(filename):  
	''' Returns a pandas frame derived from the EFlowTrack objects in the given ROOT file'''
	return getPandasSpecificObject(filename, "EFlowTrack", mass=0, charge=0, pid=89, XT="PT", Dxy_Ehad_Eem="Dxy")

def getPandasAll(filename, cull=False, includePuppi=False):
	''' Returns a pandas frame derived from several observable objects in the given ROOT file'''
	lst = [getPandasPhotons(filename),
					getPandasElectrons(filename),
					getPandasTightMuons(filename),
					getPandasEFlowParticle(filename, name="EFlowNeutralHadron"),
					getPandasEFlowParticle(filename, name="EFlowPhoton"),
					getPandasEFlowTrack(filename),
					getPandasMissingET(filename)]
	
	if(includePuppi):
		lst = lst + [getPandasMissingET(filename, "PuppiMissingET")]
	#Merge all these frames
	out = pd.concat(lst)
	return out

	

