import os
import sys
if __package__ is None:
    sys.path.append(os.path.realpath("../"))
import ROOT 
import numpy as np
import math
import time
import pandas as pd
from itertools import islice
import glob
import ntpath
import getopt
from CMS_SURF_2016.utils.meta import msgpack_assertMeta


def DeltaRsq(A_Eta, A_Phi, B_Eta, B_Phi):
    '''Computes the 
        #Arguments
            #consider N = # of A, M = # of B
            A_Eta -- A numpy array of shape (N,) of all the Eta values of a certain
                        type of object in a given sample
            A_Phi -- A numpy array of shape (N,) of all the Phi values of a certain
                        type of object in a given sample
            B_Eta -- A numpy array of shape (M,) of all the Eta values of a certain
                        type of object in a given sample
            B_Phi -- A numpy array of shape (M,) of all the Phi values of a certain
                        type of object in a given sample
        #Returns
            A numpy ndarray of shape (N,M) containing the pairwise squared angular distances  
            between two sets of objects of types A and B
    '''

    B_Eta = B_Eta.reshape(1,B_Eta.shape[-1])
    B_Phi = B_Phi.reshape(1,B_Phi.shape[-1])

    A_Eta = A_Eta.reshape(A_Eta.shape[-1],1)
    A_Phi = A_Phi.reshape(A_Phi.shape[-1],1)

    #print(A_Eta.shape, B_Eta.shape)
    DeltaEta = A_Eta - B_Eta
    DeltaPhi = A_Phi - B_Phi
    
    tooLarge = -2.*math.pi*(DeltaPhi > math.pi)
    tooSmall = 2.*math.pi*(DeltaPhi < -math.pi)
    DeltaPhi = DeltaPhi + tooLarge + tooSmall    
    delRsq = DeltaEta*DeltaEta+DeltaPhi*DeltaPhi
    return delRsq

def trackMatch(prtEta, prtPhi, trkEta, trkPhi):
    '''Matches a reconstructed particle with its track
        #Arguments
            #consider N = # of particles, M = # of tracks
            prtEta -- A numpy array of shape (N,) of all the Eta values of a certain
                        type of reconstructed particle in a given sample
            prtPhi -- A numpy array of shape (N,) of all the Phi values of a certain
                        type of reconstructed particle in a given sample
            trkEta -- A numpy array of shape (M,) of all the Eta values of a track in 
                        a given sample
            trkPhi -- A numpy array of shape (M,) of all the trkPhi values of a track in 
                        a given sample
        #Returns
            A numpy array of shape (N,) containing the index of each track corresponding to each
            reconstruced particle.
    '''
    delRsq = DeltaRsq(prtEta, prtPhi, trkEta, trkPhi)
    index = np.argmin(delRsq, axis=1)
    return index


def Iso(A_Eta, A_Phi, A_Pt, B_Eta, B_Phi,maxdist=0.3):
    '''Computes the isolation between two object types
        #Arguments
            #consider N = # of particles in group A, M = # of particles in group B
            (A or B)_Eta --  numpy array with shape (N or M,) 
                                corresponding to the Eta values of the particles
            (A or B)_Phi --  numpy array with shape (N or M,) 
                                corresponding to the Phi values of the particles
            (A or B)_Pt --  numpy array with shape (N or M,) 
                                corresponding to the transverse momentum values of the particles
            maxdist -- The maximum cartesian distance between Eta and Phi to be included in the isolation
        #Returns
            The isolations of each particle in B for each particle in A
    '''
    DRsq = DeltaRsq(A_Eta, A_Phi, B_Eta, B_Phi)

    #Exclude particles in B that are too far away
    CloseTracks = DRsq < maxdist*maxdist
    DRsq = DRsq * CloseTracks

    out = np.sum(DRsq, axis=1, dtype='float64')/A_Pt
    return out

def fill_object(dicts_by_object,leaves_by_object,entry, start_index,obj, PT_ET_MET, M, others):
    '''Fills an object with values for a given entry
        #Arguments
            dicts_by_object -- A dictionary keyed by object type containing dictionaries of arrays
                                keyed by observable type. Arrays are expected to be prefilled with
                                zeros
            leaves_by_object -- A dictionary keyed by object type, containing dictionaries of tuples 
                                like (leaf, branch) keyed by observable type. Only valid ROOT 
                                observables are used as keys. Each (leaf,branch) pair corresponds
                                to the leaf and branch of a ROOT observable.
            entry -- The entry to in the ROOT file to read from
            start_index -- Where to start filling each array in the dictionary dicts_by_object[obj].
                            They should all have the same length since each array is a column in a table.
            obj -- The type of object we are filling
            PT_ET_MET -- The particluar flavor of transverse energy for this kind of object either 
                        ('PT',  'ET', 'MET')
            M -- The mass of this kind of object
            others -- Any ROOT observables unique to this kind of object that we should read
                                             
        #Returns 
            The number of values filled in for each column of our table
            
    '''
    d = leaves_by_object[obj]
    fill_dict = dicts_by_object[obj]
    for (leaf, branch) in d.values():
        branch.GetEntry(entry)
    n_values = d["Phi"][0].GetLen()
    # print(obj,n_values)

    #start_index = index_by_objects[obj]
    lv = ROOT.TLorentzVector()
    l_PT = d[PT_ET_MET][0]
    l_Eta = d["Eta"][0]
    l_Phi = d["Phi"][0]
    l_others = [(other, d[other][0]) for other in others]
    #lv_leaves = [d[ lorentz_vars[i] ][0] for i in range(lorentz_vars)]
    for i in range(n_values):
        index = start_index + i
        # print(obj,index)
        PT = l_PT.GetValue(i)
        Eta = l_Eta.GetValue(i)
        Phi = l_Phi.GetValue(i)
        lv.SetPtEtaPhiM(PT,Eta,Phi, M)
        fill_dict["Entry"][index] = entry
        fill_dict["E/c"][index] = lv.E()
        fill_dict["Px"][index] = lv.Px()
        fill_dict["Py"][index] = lv.Py()
        fill_dict["Pz"][index] = lv.Pz()
        fill_dict["PT_ET"][index] = PT
        fill_dict["Eta"][index] = Eta
        fill_dict["Phi"][index] = Phi
        #print(Phi)
        for (other, l_other) in l_others:
            #if(obj == "EFlowTrack"): print(other, l_other.GetValue(i))
            fill_dict[other][index] = l_other.GetValue(i)
        # for OUTPUT_OBSERVS
    return n_values 
def fillTrackMatch(dicts_by_object,obj, trackIndicies, prtStart, trackStart):
    '''Fills an object with values for a given entry
        #Arguments
            dicts_by_object -- A dictionary keyed by object type containing dictionaries of arrays
                                keyed by observable type. Arrays are expected to be prefilled with
                                zeros
            obj -- The type of object we are matching with a track
            trackIndicies -- The result of trackMatch(). A numpy array of indicies corresponding to
                            tracks being matched to particles
            prtStart -- Where to start filling in X,Y,Z, and Dxy values for paricles matched to tracks
            trackStart -- Where to start reading track values from.
                #Note: We need to know where to start because we may not be filling or reading from
                        the beginning of each of our table columns.
                                             
        #Returns(void)
            
    '''
    l = dicts_by_object[obj]
    t = dicts_by_object["EFlowTrack"]
    lX = l["X"]
    lY = l["Y"]
    lZ = l["Z"]
    lDxy = l["Dxy"]
    tX = t["X"]
    tY = t["Y"]
    tZ = t["Z"]
    tDxy = t["Dxy"]
    for lepI, trkI in enumerate(trackIndicies):
        lepIndex = lepI + prtStart
        trkIndex = trkI + trackStart
        # print(obj, lepIndex, trkIndex, tX[trkIndex], tY[trkIndex], tZ[trkIndex])
        lX[lepIndex] = tX[trkIndex]
        lY[lepIndex] = tY[trkIndex]
        lZ[lepIndex] = tZ[trkIndex]
        lDxy[lepIndex] = tDxy[trkIndex]

    
def fillIso(dicts_by_object,obj, isoType,  obj_start,iso):
    '''Fills an object with values for a given entry
        #Arguments
            dicts_by_object -- A dictionary keyed by object type containing dictionaries of arrays
                                keyed by observable type. Arrays are expected to be prefilled with
                                zeros
            obj -- The type of object are running isolation on
            isoType -- The object type corresponding to the type of isolation we are running
            obj_start -- Where to start filling in isolation values
                 #Note: We need to know where to start because we may not be filling or reading from
                        the beginning of each of our table columns.
            iso -- The result of Iso(). A list of isolation values.
        #Returns(void)     
    '''
    isoArr = dicts_by_object[obj][isoType]
    # print(type(iso), iso)
    for i in range(len(iso)):
        isoArr[obj_start+i] = iso[i]

def getEtaPhiPTasNumpy(dicts_by_object,obj, start, n_vals):
    '''Gets numpy arrays corresponding to all of the Eta, Phi, and Pt values for a given object type
            in a given sample. Here start and n_vals specifiy where the sample starts and how long it is. 
        #Arguments
            dicts_by_object -- A dictionary keyed by object type containing dictionaries of arrays
                                keyed by observable type. Arrays are expected to be prefilled with
                                zeros
            obj -- The type of object we are reading from to create our numpy arrays
            start -- where to start reading in that object
                #Note: We need to know where to start because we may not be filling or reading from
                        the beginning of each of our table columns.
            n_vals -- How many values to read.
        #Returns
            Eta, Phi, Pt -- numpy arrays    
    '''
    d = dicts_by_object[obj]
    Eta =  np.array( d["Eta"][start:start+n_vals] )
    Phi =  np.array( d["Phi"][start:start+n_vals] )
    Pt =  np.array( d["PT_ET"][start:start+n_vals] )
    return Eta, Phi, Pt 


#Masses for electrons and muons
mass_of_electron = np.float64(0.0005109989461) #eV/c
mass_of_muon = np.float64(0.1056583715) 

OBJECT_TYPES = ['Electron', 'MuonTight', 'Photon', 'MissingET', 'EFlowPhoton', 'EFlowNeutralHadron', 'EFlowTrack']
PT_ET_TYPES  = ['PT',          'PT',       'PT',      'MET',        'ET',           'ET',               'PT', ]
EXTRA_FILLS  = [['Charge'], ['Charge'],     [],        [],     ['Ehad', 'Eem'],  ['Ehad', 'Eem'], ['Charge','X', 'Y', 'Z', 'Dxy'],]
MASSES =    [mass_of_electron, mass_of_muon, 0,        0,           0,                0,                 0]
TRACK_MATCH =   [True,        True,        False,    False,        False,           False,              False]
COMPUTE_ISO =   [True,        True,        True,     False,        True,           True,              False]

ROOT_OBSERVS =  ['PT', 'ET', 'MET', 'Eta', 'Phi', 'Charge', 'X', 'Y', 'Z', 'Dxy', 'Ehad', 'Eem']
OUTPUT_OBSERVS =  ['Entry','E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',\
                     'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']
ISO_TYPES = [('MuIso', 'MuonTight'), ('EleIso','Electron'), ('ChHadIso','EFlowTrack') ,('NeuHadIso','EFlowNeutralHadron'),('GammaIso','EFlowPhoton')]

def delphes_to_pandas(filepath, verbosity=1):
    start_time = time.clock()
    fileIN = ROOT.TFile.Open(filepath)
    tree = fileIN.Get("Delphes")
    n_entries=tree.GetEntries()

    tree.SetCacheSize(30*1024*1024)


    #Get all the leaves that we need to read and their associated branches
    leaves_by_object = {}
    for obj in OBJECT_TYPES:
        leaves_by_object[obj] = {}
        for observ in ROOT_OBSERVS:
            leaf = tree.GetLeaf(obj + '.' + observ)
            if(isinstance(leaf,ROOT.TLeafElement)):
                leaves_by_object[obj][observ] = (leaf, leaf.GetBranch())
                #print(leaf.GetBranch())


    #Allocate the data for the tables by filling arrays with zeros
    dicts_by_object = {}
    dicts_by_object["NumValues"] = {}
    for obj in OBJECT_TYPES:
        dicts_by_object[obj] = {}
        # print(obj)
        (leaf, branch) = leaves_by_object[obj]['Phi']
        total_values = 0

        #Loop over all the Phi values (since everything has Phi) and accumilate
        #   the total number of values for each object type
        for entry in range(n_entries):
            branch.GetEntry(entry)
            total_values += leaf.GetLen()

        #Fill arrays with zeros to avoid reallocating data later
        for observ in OUTPUT_OBSERVS:
            dicts_by_object[obj][observ] = [0] * total_values
        dicts_by_object["NumValues"][obj] = [0] * n_entries
        # print(total_values)
    

    index_by_objects = {o:0 for o in OBJECT_TYPES}
    last_time = time.clock()
    prev_entry = 0
    for entry in range(n_entries):

        #Make a pretty progress bar in the terminal
        if(verbosity > 0):
            c = time.clock() 
            if(c > last_time + .25):
                percent = float(entry)/float(n_entries)
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %r/%r  %r(Entry/sec)" % ('='*int(20*percent), entry, int(n_entries), 4 * (entry-prev_entry)))
                sys.stdout.flush()
                last_time = c
                prev_entry = entry

        #Initialize some temporary helper variables
        number_by_object = {}
        Eta_Phi_PT_by_object = {}

        #Fill each type of object with everything that is observable in the ROOT file for that object
        #   in addition to Energy and the three components of momentum
        for obj, PT_ET_type, mass, extra_fills in zip(OBJECT_TYPES, PT_ET_TYPES, MASSES, EXTRA_FILLS):
            start = index_by_objects[obj]
            n = fill_object(dicts_by_object,leaves_by_object,entry, start, obj, PT_ET_type, mass, extra_fills)
            dicts_by_object["NumValues"][obj][entry] = n
            number_by_object[obj] = n
            Eta_Phi_PT_by_object[obj] = getEtaPhiPTasNumpy(dicts_by_object,obj, start, n)

        #Do Track matching for objects with TRACK_MATCH = Trie
        trkEta, trkPhi, dummy = Eta_Phi_PT_by_object["EFlowTrack"]
        start_tracks = index_by_objects["EFlowTrack"]
        for obj, ok in zip(OBJECT_TYPES, TRACK_MATCH):
            if(ok):
                start = index_by_objects[obj]
                Phi, Eta, PT = Eta_Phi_PT_by_object[obj]
                matches = trackMatch(Phi, Eta, trkEta, trkPhi)
                fillTrackMatch(dicts_by_object,obj, matches, start, start_tracks)

        #Compute isolation
        for obj, ok in zip(OBJECT_TYPES, COMPUTE_ISO):
            start = index_by_objects[obj]
            if(ok):
                objEta, objPhi, objPt = Eta_Phi_PT_by_object[obj]
                for iso_type, iso_obj in ISO_TYPES:
                    isoEta, isoPhi, isoPt = Eta_Phi_PT_by_object[iso_obj]
                    iso_val = Iso(objEta, objPhi, objPt, isoEta, isoPhi) 
                    iso_val = iso_val - 1.0 if obj == iso_obj else iso_val
                    fillIso(dicts_by_object,obj, iso_type,  start, iso_val)

        for obj in OBJECT_TYPES:
            index_by_objects[obj] += number_by_object[obj]
    pandas_out = {}
    for obj,d in dicts_by_object.items():
        if(obj == "NumValues"):
            pandas_out[obj] = pd.DataFrame(d, columns=OBJECT_TYPES)
            #print(pandas_out[obj])
        else:
            pandas_out[obj] = pd.DataFrame(d, columns=OUTPUT_OBSERVS)
        # if(TRACK_MATCH[OBJECT_TYPES.index(obj)] or obj == "EFlowTrack"):
        # if(COMPUTE_ISO[OBJECT_TYPES.index(obj)]):
            # print(obj)
            # print(pandas_out[obj][['MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']])

    print("ElapseTime: %.2f" % float(time.clock()-start_time))

    return pandas_out


#http://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion
def roundrobin(*iterables):
   "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
   # Recipe credited to George Sakkis
   pending = len(iterables)
   nexts = cycle(iter(it).next for it in iterables)
   while pending:
      try:
         for next in nexts:
             yield next()
      except StopIteration:
         pending -= 1
         nexts = cycle(islice(nexts, pending))




def makeJobs(filename,
                storeType,
                directory="/data/shared/Delphes/",
                folder="/pandas/"):
    if(filename[len(filename)-1] == "/"): filename = filename[0:-1]
    files = glob.glob(directory + filename + "/*.root")
    store_dir = directory + filename + folder
    print(store_dir)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    jobs = [ (f,  store_dir, storeType) for f in files]
    # for f in files:
    #    f_name =
    #    jobs.append((f, store_dir))
    return jobs

def doJob(job, redo=False):
    f, store_dir, storeType = job
    try:
        store(f, store_dir,rerun=redo,storeType=storeType)
    except Exception as e:
        print(e)
        print("Failed to parse file %r. File may be corrupted." % f)
    return f




def store(filepath, outputdir, rerun=False, storeType="hdf5"):
    filename = os.path.splitext(ntpath.basename(filepath))[0]
    if(storeType == "hdf5"):
        out_file = outputdir + filename + ".h5"
        print(out_file)
        store = pd.HDFStore(out_file)
        keys = store.keys()
        #print("KEYS:", set(keys))
        #print("KEYS:", set(["/"+key for key in OBJECT_TYPES+["NumValues"]]))
        #print("KEYS:", set(keys)==set(["/"+key for key in OBJECT_TYPES+["NumValues"]]))
        if(set(keys) != set(["/"+key for key in OBJECT_TYPES+["NumValues"]]) or rerun):
            #print("OUT",out_file)
            frames = delphes_to_pandas(filepath)
            for key,frame in frames.items():
                store.put(key, frame, format='table')
        store.close()
    elif(storeType == "msgpack"):
        out_file = outputdir + filename + ".msg"
        # meta_out_file = outputdir + filename + ".meta"
        print(out_file)
        if(not os.path.exists(out_file) or rerun):
            frames = delphes_to_pandas(filepath)
            # meta_frames = {"NumValues" : frames["NumValues"]}
            pd.to_msgpack(out_file, frames)
            # pd.to_msgpack(meta_out_file, meta_frames)
            msgpack_assertMeta(out_file, frames)
        else:
            msgpack_assertMeta(out_file)
        # elif(not os.path.exists(meta_out_file)):
        #     print(".meta file missing creating %r" % meta_out_file)
        #     frames = pd.read_msgpack(out_file)
        #     meta_frames = {"NumValues" : frames["NumValues"]}
        #     pd.to_msgpack(meta_out_file, meta_frames)
    else:
        raise ValueError("storeType %r not recognized" % storeType)


def main(data_dir, argv):
    # print(data_dir)
    storeType = "hdf5"
    redo = False
    screwup_error = "python delphes_parse.py <input_dir>"
    try:
        opts, args = getopt.getopt(argv,'mrh')
    except getopt.GetoptError:
        print screwup_error
        sys.exit(2)
  
    for opt, arg in opts:
      # print(opt, arg)
        if opt in ("-m", "--msg", "--msgpack"):
            storeType = "msgpack"
        elif opt in ('-h5', "--hdf", "--hdf5"):
            storeType = "hdf5"
        elif opt in ('-r', "--redo"):
             redo = True
    print(storeType)
    folder = "/pandas_h5/" if storeType == "hdf5" else "/pandas_msg/"
    jobs = makeJobs(data_dir,storeType, folder=folder)
    for job in jobs:
        # print(job)
        doJob(job, redo=redo)




if __name__ == "__main__":

   main(sys.argv[1],sys.argv[2:])




# if __name__ == "__main__":
#     delphes_to_pandas("../data/ttbar_lepFilter_13TeV_147.root")


        
