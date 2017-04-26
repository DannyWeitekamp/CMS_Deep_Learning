from __future__ import absolute_import

import unittest
import sys, os
if __package__ is None:
    print(os.path.realpath("../"))
    sys.path.append(os.path.realpath("../"))
else:
    print(__package__)
import tempfile
import numpy as np
import pandas as pd
from CMS_Deep_Learning.preprocessing.preprocessing import ObjectProfile, preprocessFromPandas_label_dir_pairs,procsFrom_label_dir_pairs
gen_observ_types = ['PT_ET','Eta', 'Phi']
observ_types = gen_observ_types + ["ObjType"]

obs_pl_t = ["Entry"] + gen_observ_types
vecsize = len(obs_pl_t)
RANDOM_SEED = 7
np.random.seed(seed=RANDOM_SEED)

object_profiles1 = [ObjectProfile("EFlowPhoton",2, pre_sort_columns="PT_ET",
                                 pre_sort_ascending=False,
                                 addColumns={"ObjType":1}),
                   ObjectProfile("EFlowTracks",2, pre_sort_columns=["PT_ET"],
                                 pre_sort_ascending=True,
                                 addColumns={"ObjType":2}),
                   ObjectProfile("EFlowNeutralHadron",2, pre_sort_columns=["PT_ET"],
                                 pre_sort_ascending=False, sort_columns="Phi", sort_ascending=True,
                                 addColumns={"ObjType":3}),
                   ObjectProfile("MET",1, pre_sort_columns=["PT_ET"],
                                 pre_sort_ascending=False, sort_columns=["Phi"],
                                 addColumns={"ObjType":4}),
                   ObjectProfile("MuonTight",2, pre_sort_columns=["PT_ET"],
                                 pre_sort_ascending=False, sort_columns=["Phi"], sort_ascending=False,
                                 addColumns={"ObjType":5}),
                   ObjectProfile("Electron",2, #pre_sort_columns=["PT_ET"],
                                 pre_sort_ascending=False, sort_columns=["Phi"],
                                 addColumns={"ObjType":6})
                  ]

RAND_OPS =  [ObjectProfile("EFlowPhoton", 100, pre_sort_columns="shuffle",
                                      pre_sort_ascending=False,
                                      addColumns={"ObjType": 1}),
                        ObjectProfile("EFlowTracks", 100, pre_sort_columns=["shuffle"],
                                      pre_sort_ascending=True,
                                      addColumns={"ObjType": 2}),
                        ObjectProfile("EFlowNeutralHadron", 100, pre_sort_columns=["shuffle"],
                                      pre_sort_ascending=False, sort_columns="shuffle", sort_ascending=True,
                                      addColumns={"ObjType": 3}),
                        ObjectProfile("MET", 1, pre_sort_columns=["random"],
                                      pre_sort_ascending=False, sort_columns=["random"],
                                      addColumns={"ObjType": 4}),
                        ObjectProfile("MuonTight", 5, pre_sort_columns=["random"],
                                      pre_sort_ascending=False, sort_columns=["random"], sort_ascending=False,
                                      addColumns={"ObjType": 5}),
                        ObjectProfile("Electron", 5,  # pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=False, sort_columns=["random"],
                                      addColumns={"ObjType": 6})
                        ]

def rand_pl_entry(entry, a,b,marker=None):
    out = np.concatenate([np.full((a,1),entry, dtype='float64'), np.random.randn(a,b) if marker is None else np.full((a,b),marker)], axis = 1)
    # print(out)
    return out
def norm_uint(mean, std):
    return max(int(np.random.normal(mean, std)),0)
def fake_frames(N,object_profiles, marker=None, nb_eflow=3,std_eflow=1):
    vecsize = len(obs_pl_t)
    frames = {profile.name:pd.DataFrame(columns=obs_pl_t) for profile in object_profiles}
    #print(frames.values()[0].shape, (1,1, vecsize))
    num_val_dict = {key:[None]*N for key, frame in frames.items()}
    for profile in object_profiles:
        frames[profile.name] = pd.DataFrame(columns=obs_pl_t)
    # print("MARKER",marker)
    for entry in range(N):
        #n = norm_uint(100,35)
        n = norm_uint(nb_eflow,std_eflow)
        num_val_dict["EFlowPhoton"][entry] = n       
        frames["EFlowPhoton"] = pd.concat([frames["EFlowPhoton"],pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])
       
        #n = norm_uint(120, 23)
        n = norm_uint(nb_eflow,std_eflow)
        num_val_dict["EFlowTracks"][entry] = n  
        frames["EFlowTracks"] = pd.concat([frames["EFlowTracks"] ,pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])
        
        #n = norm_uint(90, 27)
        n = norm_uint(nb_eflow,std_eflow)
        num_val_dict["EFlowNeutralHadron"][entry] = n
        frames["EFlowNeutralHadron"] = pd.concat([frames["EFlowNeutralHadron"] ,pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])

        n = 1
        num_val_dict["MET"][entry] = n
        frames["MET"] = pd.concat([frames["MET"] ,pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])

        n = int(np.random.uniform(0, 5))
        num_val_dict["MuonTight"][entry] = n
        frames["MuonTight"] = pd.concat([frames["MuonTight"] ,pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])

        n = int(np.random.uniform(0, 5))
        num_val_dict["Electron"][entry] = n
        frames["Electron"] = pd.concat([frames["Electron"] ,pd.DataFrame(rand_pl_entry(entry,n, vecsize-1,marker), columns=obs_pl_t)])
    frames["NumValues"] = pd.DataFrame(num_val_dict)
    print("NOOOOOP",num_val_dict)
    return frames

def store_frames(frames, filepath):
    store = pd.HDFStore(filepath)
    for key,frame in frames.items():
        store.put(key, frame, format='table')
    store.close()
    

def store_fake(directory, size, num, object_profiles, marker=None, nb_eflow=3,std_eflow=1):
    if not os.path.exists(directory):
        os.makedirs(directory)
    frames_list = [None]* num
    for i in range(num):
        frames = fake_frames(size, object_profiles, marker=marker,nb_eflow=nb_eflow,std_eflow=std_eflow)
        store_frames(frames, directory+"%03i.h5" % i)
        frames_list[i] = frames
    return frames_list

temp_dir = tempfile.gettempdir() + "/fake_delphes/"
ttbar_dir = temp_dir + "ttbar/"
wjet_dir = temp_dir + "wjet/"
qcd_dir = temp_dir + "qcd/"
frame_lists = {}
label_dir_pairs = [("ttbar", ttbar_dir), ("wjet", wjet_dir), ("qcd", qcd_dir)]

import operator

def justCheckSize(t,X, Y,sizes):
    t.assertTrue(np.array_equal(np.array([x.shape for x in X]),sizes), msg="data is wrong size")
    t.assertFalse(True in [x.dtype == np.dtype(object) for x in X],  msg="Failed: data is not square")

def checkGeneralSanity(t, X, Y, frame_lists, sizes,  NUM, label_dir_pairs):
    is_single_list = False
    if(not isinstance(X, list)):
        is_single_list = True
        X = [X]
    justCheckSize(t,X, Y,sizes)

    all_values_by_label = {tup[0]:[None] * NUM for tup in label_dir_pairs}
    
    for entry in range(NUM):
        for label, frame_list in frame_lists.items():
            for f in frame_list:
                f = {k: df.query("Entry == %r" % entry) for k, df in f.items() if k != "NumValues"}
                all_values_by_label[label][entry] =pd.concat([df for df in f.values()]) 
    
    tn = [0]*5
    tn[4] = {p[0]: 0 for p in label_dir_pairs}
    z = np.zeros(vecsize)
    for i in range(1,len(sizes)+1):
        x = X[i-1]
        for s in x:
            was_zero = False
            for row in s:
                iszero = np.array_equal(row,z)
                if(not iszero):
                    if(was_zero):
                        tn[0] = 1 
                    #print(row[vecsize-1])
                    if (not row[vecsize-1] == i):
                        tn[1] = 1
                if(iszero): 
                    was_zero = True
                else:
                    row = row[:-1]
                    ok = 0
                    for label, frame in all_values_by_label.items():
                        frame = pd.concat(frame)
                        frame = frame.drop("Entry", axis=1)
                        if((frame == row).all(1).any()):
                            ok += 1
                            tn[4][label] = True
                    if(ok == 0):
                        tn[2] = 3
                    if(ok > 1):
                        tn[3] = 3

    t.assertEqual(tn[0], 0, msg="Padding not at end")
    t.assertEqual(tn[1], 0 ^ is_single_list, msg="Add column value incorrect")
    t.assertEqual(tn[2], 0, msg="Data does not come from table")
    t.assertEqual(tn[3], 0, msg="Row persists between samples of different classes")
    t.assertNotEqual(min(tn[4].values()), 0, msg="Not all labels are used")
    
def checkCutsAndSorts(t, X, Y, frame_lists, sizes,  NUM, label_dir_pairs, object_profiles, observ_types, sort_columns=None, sort_ascending=True):
    first = lambda x: x if not isinstance(x, list) else x[0]
    is_single_list = False
    if(not isinstance(X, list)):
        is_single_list = True
        if(sort_columns != None):
            object_profiles = [ObjectProfile(" ", 0, sort_columns=sort_columns, sort_ascending=sort_ascending)]
        else:
            return
        X = [X]
        
    tn = [0]* 4
    z = np.zeros(vecsize)
    
    for index, profile in enumerate(object_profiles):
        x = X[index]
        sort_index = None
        ascending = False
        try:
            if(profile.sort_columns == None):
                if(True in [c in profile.pre_sort_columns for c in ['shuffle', 'random']]):
                    sort_index = -1
                else:
                    sort_index = observ_types.index(first(profile.pre_sort_columns))
                    ascending = profile.pre_sort_ascending
                ti = 0
            else:
                if (True in [c in profile.sort_columns for c in ['shuffle', 'random']]):
                    sort_index = -1
                else:
                    sort_index = observ_types.index(first(profile.sort_columns))
                    ascending = profile.sort_ascending
                ti = 1
        except ValueError:
            pass
        if(sort_index != None):
            if(sort_index == -1):
                asc_list = np.array([1] * x.shape[2]) * (x.shape[1]-1)
                dec_list = np.array([1] * x.shape[2]) * (x.shape[1]-1)
                for s in x:
                    prev_row = None
                    for row in s:
                        iszero = np.array_equal(row, z)
                        if (not isinstance(prev_row, type(None)) and not iszero):
                            asc_list = asc_list*(row < prev_row)
                            dec_list = asc_list*(row < prev_row)
                        prev_row = row

                print(asc_list)
                print(dec_list)
                tn[ti+2] = sum(asc_list) + sum(dec_list)
            else:
                for s in x:
                    prev_row = None
                    for row in s:
                        iszero = np.array_equal(row,z)
                        if(not isinstance(prev_row, type(None)) and not iszero):
                            if(ascending):
                                if(row[sort_index] < prev_row[sort_index]):
                                    tn[ti] = 1
                            else:
                                if(row[sort_index] > prev_row[sort_index]):
                                    tn[ti] = 1    
                        prev_row = row
        t.assertEqual(tn[0], 0, msg="presorting incorrect.")
        t.assertEqual(tn[1], 0, msg="sorting incorrect.")
        t.assertEqual(tn[2], 0, msg="pre-shuffle incorrect.")
        t.assertEqual(tn[3], 0, msg="shuffle incorrect.")

def checkDuplicates(t,X, Y, object_profiles):
    is_single_list = False
    if(not isinstance(X, list)):
        is_single_list = True
        X = [X]
    z = np.zeros(vecsize)
    rows = []
    for index, profile in enumerate(object_profiles):
        x = X[index]
        for s in x:
            for row in s:
                iszero = np.array_equal(row,z)
                if(not iszero):
                    rows.append(tuple(row))
    t.assertEqual(len(set(rows)), len(rows), msg="Duplicate Found")
    

       
def speedTest():
    OPS = [ObjectProfile("EFlowPhoton", 100, pre_sort_columns="PT_ET",
                                      pre_sort_ascending=False,
                                      addColumns={"ObjType": 1}),
                        ObjectProfile("EFlowTracks", 100, pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=True,
                                      addColumns={"ObjType": 2}),
                        ObjectProfile("EFlowNeutralHadron", 100, pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=False, sort_columns="Phi", sort_ascending=True,
                                      addColumns={"ObjType": 3}),
                        ObjectProfile("MET", 1, pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=False, sort_columns=["Phi"],
                                      addColumns={"ObjType": 4}),
                        ObjectProfile("MuonTight", 5, pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=False, sort_columns=["Phi"], sort_ascending=False,
                                      addColumns={"ObjType": 5}),
                        ObjectProfile("Electron", 5,  # pre_sort_columns=["PT_ET"],
                                      pre_sort_ascending=False, sort_columns=["Phi"],
                                      addColumns={"ObjType": 6})
                        ]
    
    NUM = 10000
    if(os.path.getsize(temp_dir + "qcd/000.h5") < 14738451):
        frame_lists = {l: store_fake(d, NUM/10, 10, object_profiles1, nb_eflow=120, std_eflow=40) for l, d in label_dir_pairs}

    X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs, 0, NUM, OPS, observ_types, verbose=1)


class PreprocessingTests(unittest.TestCase):

    def test_normal(self):
        NUM = 2
        frame_lists = {l:store_fake(d,NUM, 1, object_profiles1) for l, d in label_dir_pairs}
        OPS = object_profiles1
        
        #SORTED
        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,0, NUM, OPS, observ_types, verbose=1)
        # print([x.shape for x in X], Y.shape)
        sizes = np.array([[len(label_dir_pairs)*NUM, p.max_size, vecsize] for p in OPS])
        checkGeneralSanity(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs)
        checkCutsAndSorts(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs, OPS, observ_types)

        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,0, NUM, OPS, observ_types, verbose=1, single_list=True)
        # print(X.shape, Y.shape)
        sizes = np.array([[len(label_dir_pairs)*NUM, sum([p.max_size for p in OPS]), vecsize]])
        checkGeneralSanity(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs)
        checkCutsAndSorts(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs, OPS, observ_types)
        
        #RANDOM SHUFFLES
        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs, 0, NUM, RAND_OPS, observ_types, verbose=1)
        # print([x.shape for x in X], Y.shape)
        sizes = np.array([[len(label_dir_pairs) * NUM, p.max_size, vecsize] for p in RAND_OPS])
        checkGeneralSanity(self, X, Y, frame_lists, sizes, NUM, label_dir_pairs)
        checkCutsAndSorts(self, X, Y, frame_lists, sizes, NUM, label_dir_pairs, RAND_OPS, observ_types)

        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs, 0, NUM, RAND_OPS, observ_types, verbose=1,
                                                    single_list=True)
        # print(X.shape, Y.shape)
        sizes = np.array([[len(label_dir_pairs) * NUM, sum([p.max_size for p in RAND_OPS]), vecsize]])
        checkGeneralSanity(self, X, Y, frame_lists, sizes, NUM, label_dir_pairs)
        checkCutsAndSorts(self, X, Y, frame_lists, sizes, NUM, label_dir_pairs, RAND_OPS, observ_types)
        

        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,0, NUM, OPS, observ_types, verbose=1, single_list=True,
                                                    sort_columns="Eta", sort_ascending=False)
        checkCutsAndSorts(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs, OPS, observ_types,
                         sort_columns="Eta", sort_ascending=False)
        # print(Y)
        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,0, NUM, OPS, observ_types, verbose=1, single_list=True,
                                                    sort_columns=["Phi"], sort_ascending=True)
        checkCutsAndSorts(self,X, Y, frame_lists, sizes,  NUM, label_dir_pairs, OPS, observ_types,
                          sort_columns=["Phi"], sort_ascending=True)

    def test_multiple_files(self):
        NUM = 15
        OPS = object_profiles1
        frame_lists = {l:store_fake(d,5, 4, object_profiles1) for l, d in label_dir_pairs}

        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,5, NUM, OPS, observ_types, verbose=1)
        sizes = np.array([[len(label_dir_pairs)*NUM, p.max_size, vecsize] for p in OPS])
        justCheckSize(self,X,Y, sizes)
        checkDuplicates(self,X,Y,OPS)

    def test_procsFrom_label_dir_pairs(self):
        NUM = 20
        frame_lists = {l: store_fake(d, 5, 4, object_profiles1) for l, d in label_dir_pairs}
        OPS = object_profiles1

        l = procsFrom_label_dir_pairs(start=0,
                                        samples_per_label=20,
                                        stride=4,
                                        archive_dir=temp_dir+"keras_archive/",
                                        label_dir_pairs=label_dir_pairs,
                                        object_profiles=OPS,
                                        observ_types=observ_types,
                                        single_list=False,
                                        sort_columns=None,
                                        sort_ascending=True,
                                        verbose=1)
        self.assertEqual(len(l), 5)

    def test_shuffle(self):
        NUM = 20
        frame_lists = {l: store_fake(d, NUM, 1, object_profiles1, marker=i) for i,(l, d) in enumerate(label_dir_pairs)}
        OPS = object_profiles1
        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs,0, NUM, OPS, observ_types, verbose=1)

        #Use MET because there is 0 chance that it does not exist
        x_check, y_check = np.array([int(x[0][0]) for x in X[3]]), np.array([y.tolist().index(1.0) for y in Y])
        # print(zip(x_check, y_check))
        self.assertTrue(np.array_equal(x_check, y_check))

        X, Y = preprocessFromPandas_label_dir_pairs(label_dir_pairs, 0, NUM, OPS, observ_types, verbose=1,
                                                    single_list=True,sort_columns=["Phi"], sort_ascending=True)
        x_check, y_check = np.array([int(x[0][0]) for x in X]), np.array([y.tolist().index(1.0) for y in Y])
        # print(zip(x_check, y_check))
        self.assertTrue(np.array_equal(x_check, y_check))

if __name__ == '__main__':
    # unittest.main()
    speedTest()



