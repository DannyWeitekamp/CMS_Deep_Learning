import numpy as np
import pandas as pd
import glob


class ObjectProfile():
	def __init__(self, name, max_size=100, sort_columns=None, sort_ascending=True, query=None, shuffle=False):
		''' Processing instructions for each type of our data types
			#Arguements:
				name	   -- The name of the data type (i.e. Electron, Photon, EFlowTrack, etc.)
				max_size   -- The maximum number of objects to use in training
				sort_columns -- What columns to sort on (See pandas.DataFrame.sort)
				sort_ascending -- Whether each column will be sorted ascending or decending (See pandas.DataFrame.sort)
				query		-- A selection query string to use before truncating the data (See pands.DataFrame.query)
				shuffle 	-- Whether or not to shuffle the data

		'''
		self.name = name
		self.max_size = max_size
		self.sort_columns = sort_columns
		self.sort_ascending = sort_ascending
		self.query = query
		self.shuffle = shuffle


	def __str__(self):
		main_clause = "name:%r max_size=%r " % (self.name, self.max_size)
		sort_clause = ""
		query_clause = ""
		if(self.sort_columns != None):
			sort_clause = "sort_columns=%r sort_ascending=%r " % (self.sort_columns, self.sort_ascending)
		if(self.query != None):
			query_clause = "query=%r " % (self.query)
		shuffle_clause = "shuffle=%r" % self.shuffle

		return main_clause + sort_clause + query_clause + shuffle_clause
	
	__repr__ = __str__

def padItem(x,max_size, vecsize, shuffle=False):
	'''Pads a numpy array up to MAX_SIZE or trucates it down to MAX_SIZE. Shuffle, shuffles the padded output before returning'''
	if(len(x) > max_size):
		out = x[:max_size]
	else:
		out = np.append(x ,np.array(np.zeros((max_size - len(x), vecsize))), axis=0)
	if(shuffle == True): np.random.shuffle(out)
	return out
   
	#arr[index] = np.array(padItem(x.values, max_size, shuffle=shuffle))
def preprocessFromPandas_label_dir_pairs(label_dir_pairs,num_samples, object_profiles, observ_types):
	X_train = {}
	y_train = []
	X_train_indices = {}
	
	vecsize = len(observ_types)
	num_labels = len(label_dir_pairs)

	#Build vectors in the form [1,0,0], [0,1,0], [0, 0, 1] corresponding to each label
	label_vecs = {}
	for i, (label, data_dir) in enumerate(label_dir_pairs):
		arr = np.zeros((num_labels,))
		arr[i] = 1
		label_vecs[label] = arr

	print(label_vecs)
	
	#Prefil; the arrays so that we don't waste time resizing lists
	for profile in object_profiles:
		key = profile.name
		X_train[key] = [None] * (num_samples * num_labels)
		X_train_indices[key] = 0
	y_train = [None] * (num_samples * num_labels)
	
	y_train_start = 0
	for (label,data_dir) in label_dir_pairs:
		files = glob.glob(data_dir+"*.h5")
		samples_read = 0
		for f in files:
		  
            #Get the HDF Store for the file
			store = pd.HDFStore(f)

            #Get the NumValues frame which lists the number of values for each entry
			num_val_frame = store.get('/NumValues')
			max_entries = len(num_val_frame.index)
			samples_to_read = min(num_samples-samples_read, max_entries)
			num_val_frame = num_val_frame[:samples_to_read]
			assert samples_to_read >= 0
			
			
			print("Reading %r samples from %r:" % (samples_to_read,f))
			
			
			for profile in object_profiles:
				key = profile.name
				max_size = profile.max_size
				print("Mapping %r Values/Sample from %r" % (max_size, key))
				nums = num_val_frame[key]
				stop = nums.sum()
				
				if(samples_to_read == max_entries):
					frame = store.get('/'+key)
				else:
					frame = store.select('/'+key, start=0, stop=stop)
			   
				
				start = X_train_indices[key]
				arr = X_train[key]

				#Group by Entry
				groups = frame.groupby(["Entry"], group_keys=True)#[observ_types]
				group_itr = iter(groups)
				
				#Go through the all of the groups by entry and apply preprocessing based off of the object profile
				#TODO: is a strait loop slow? Should I use apply(lambda...etc) instead? Is that possible if I need to loop
				#	  over index, x and not just x?
				for entry, x in group_itr:
					if(profile.query != None):
						x = x.query(profile.query)
					if(profile.sort_columns != None):
						x = x.sort(profile.sort_columns, ascending=profile.sort_ascending)
					x = padItem(x[observ_types].values, max_size, vecsize, shuffle=profile.shuffle)
					arr[start + entry] = x
				
				#Go through the all of the entries that were empty for this datatype and make sure we pad them
				for i in range(start, start+samples_to_read):
					if(arr[i] == None):
						arr[i] = np.array(np.zeros((max_size, vecsize)))
				X_train_indices[key] += samples_to_read
				frame = None
				groups = None
			
			num_val_frame = None
			store.close()
			samples_read += samples_to_read
			print(samples_read, num_samples)
			if(samples_read >= num_samples):
				assert samples_read == num_samples
				break
		
		
		
		for i in range(num_samples):
			y_train[y_train_start+i] = label_vecs[label]
		y_train_start += num_samples
	
	y_train = np.array(y_train)
	
	indices = np.arange(len(y_train))
	np.random.shuffle(indices)
	for key in X_train:
		X_train[key] = np.array(X_train[key])[indices]

	y_train = y_train[indices]
	return X_train, y_train