import os
import sys 
import ROOT 
ROOT.gSystem.Load("libDelphes.so")
#from ROOT import Tower
#from ROOT import Muon
#from ROOT import Electron
#from ROOT import Track
import numpy as np
import math
import time
import pandas as pd
from itertools import islice

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
    Eta =  np.array( dicts_by_object[obj]["Eta"][start:start+n_vals] )
    Phi =  np.array( dicts_by_object[obj]["Phi"][start:start+n_vals] )
    Pt =  np.array( dicts_by_object[obj]["PT_ET"][start:start+n_vals] )
    return Eta, Phi, Pt 


#Masses for electrons and muons
mass_of_electron = np.float64(0.0005109989461) #eV/c
mass_of_muon = np.float64(0.1056583715) 

def Convert(verbosity=1):
    start_time = time.clock()
    fileIN = ROOT.TFile.Open("data/ttbar_lepFilter_13TeV_147.root")
    tree = fileIN.Get("Delphes")
    n_entries=10#tree.GetEntries()

    tree.SetCacheSize(30*1024*1024)


    OBJECT_TYPES = ['Electron', 'MuonTight', 'Photon', 'MissingET', 'EFlowPhoton', 'EFlowNeutralHadron', 'EFlowTrack']
    PT_ET_TYPES  = ['PT',          'PT',       'PT',      'MET',        'ET',           'ET',               'PT', ]
    EXTRA_FILLS  = [['Charge'], ['Charge'],     [],        [],     ['Ehad', 'Eem'],  ['Ehad', 'Eem'], ['Charge','X', 'Y', 'Z', 'Dxy'],]
    MASSES =    [mass_of_electron, mass_of_muon, 0,        0,           0,                0,                 0]
    TRACK_MATCH =   [True,        True,        False,    False,        False,           False,              False]
    COMPUTE_ISO =   [True,        True,        True,     False,        False,           True,              False]

    ROOT_OBSERVS =  ['PT', 'ET', 'MET', 'Eta', 'Phi', 'Charge', 'X', 'Y', 'Z', 'Dxy', 'Ehad', 'Eem']
    OUTPUT_OBSERVS =  ['Entry','E/c', 'Px', 'Py', 'Pz', 'PT_ET','Eta', 'Phi', 'Charge', 'X', 'Y', 'Z',\
                         'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']
    ISO_TYPES = [('MuIso', 'MuonTight'), ('EleIso','Electron'), ('ChHadIso','EFlowTrack') ,('NeuHadIso','EFlowNeutralHadron'),('GammaIso','Photon')]


    #Get all the leaves that we need to read and their associated branches
    leaves_by_object = {}
    for obj in OBJECT_TYPES:
        leaves_by_object[obj] = {}
        for observ in ROOT_OBSERVS:
            leaf = tree.GetLeaf(obj + '.' + observ)
            if(isinstance(leaf,ROOT.TLeafElement)):
                leaves_by_object[obj][observ] = (leaf, leaf.GetBranch())
                print(leaf.GetBranch())


    #Allocate the data for the tables by filling arrays with zeros
    dicts_by_object = {}
    for obj in OBJECT_TYPES:
        dicts_by_object[obj] = {}
        print(obj)
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
        print(total_values)
    

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
            number_by_object[obj] = n
            Eta_Phi_PT_by_object[obj] = getEtaPhiPTasNumpy(dicts_by_object,obj, start, n)

        #Do Track matching for objects designated for it.
        trkEta, trkPhi, dummy = Eta_Phi_PT_by_object["EFlowTrack"]
        start_tracks = index_by_objects["EFlowTrack"]
        for obj, ok in zip(OBJECT_TYPES, TRACK_MATCH):
            if(ok):
                start = index_by_objects[obj]
                Phi, Eta, PT = Eta_Phi_PT_by_object[obj]
                matches = trackMatch(Phi, Eta, trkEta, trkPhi)
                fillTrackMatch(dicts_by_object,obj, matches, start, start_tracks)

        for obj, ok in zip(OBJECT_TYPES, COMPUTE_ISO):
            start = index_by_objects[obj]
            if(ok):
                objEta, objPhi, objPt = Eta_Phi_PT_by_object[obj]
                for iso_type, iso_obj in ISO_TYPES:
                    isoEta, isoPhi, isoPt = Eta_Phi_PT_by_object[iso_obj]
                    # print(Iso)
                    iso_val = Iso(objEta, objPhi, objPt, isoEta, isoPhi) 
                    iso_val = iso_val - 1.0 if obj == iso_obj else iso_val
                    # print(obj, iso_val.shape)
                    fillIso(dicts_by_object,obj, iso_type,  start, iso_val)

        for obj in OBJECT_TYPES:
            index_by_objects[obj] += number_by_object[obj]
    pandas_out = {}
    for obj in OBJECT_TYPES:
        d = dicts_by_object[obj]
        pandas_out[obj] = pd.DataFrame(d, columns=OUTPUT_OBSERVS)
        # if(TRACK_MATCH[OBJECT_TYPES.index(obj)] or obj == "EFlowTrack"):
        if(COMPUTE_ISO[OBJECT_TYPES.index(obj)]):
            print(obj)
            print(pandas_out[obj][['MuIso', 'EleIso', 'ChHadIso','NeuHadIso','GammaIso']])

    print("ElapseTime: %.2f" % float(time.clock()-start_time))

    return pandas_out

if __name__ == "__main__":
    Convert()
    

           # print(pandas_out[obj][['Eta','Phi','X',  'Y',  'Z',  'Dxy'] ])

        # index_by_objects["Electron"] += n_electrons
        # index_by_objects["Photon"] += n_photons    
        # index_by_objects["MissingET"] += n_missingET
        # index_by_objects["EFlowTrack"] += n_tracks
        # index_by_objects["EFlowPhoton"] += n_ePhotons
        # index_by_objects["EFlowNeutralHadron"] += n_eHadrons

        # start_tracks = index_by_objects["EFlowTrack"] 
        # n_tracks = fill_object(entry, start_tracks, "EFlowTrack", "PT", 0, ["X","Y","Z","Dxy"])
        # trkEta, trkPhi, trkPt = getEtaPhiPTasNumpy("EFlowTrack", start_tracks, n_tracks)


        # start_muons = index_by_objects["MuonTight"] 
        # n_muons = fill_object(entry, start_muons, "MuonTight", "PT", 0, ["Charge"])
        # muEta, muPhi, muPt = getEtaPhiPTasNumpy("MuonTight", start_muons, n_muons)
        # matches = trackMatch(muEta, muPhi, trkEta, trkPhi)
        # fillTrackMatch("MuonTight", matches, start_muons, start_tracks)


        # start_electrons = index_by_objects["Electron"] 
        # n_electrons = fill_object(entry, start_electrons, "Electron", "PT", 0, ["Charge"])
        # eleEta, elePhi, elePt = getEtaPhiPTasNumpy("Electron", start, n_electrons)
        # matches = trackMatch(eleEta, elePhi, trkEta, trkPhi)
        # fillTrackMatch("Electron", matches, start_electrons, start_tracks)

        # start_gamma = index_by_objects["Photon"] 
        # n_photons = fill_object(entry, start_gamma, "Photon", "PT", 0, [])
        # gammaEta, gammaPhi, gammaPt = getEtaPhiPTasNumpy("Photon", start_gamma, n_photons)
       

        # print("ETA",gammaEta)
        # print("ETA",muEta)
        # print("ETA",eleEta)

        # start_missingET = index_by_objects["MissingET"] 
        # n_missingET = fill_object(entry, start_missingET, "MissingET", "MET", 0, [])
        # # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))

        # start_ePhoton = index_by_objects["EFlowPhoton"] 
        # n_ePhotons = fill_object(entry, start_ePhoton, "EFlowPhoton", "ET", 0, ["Ehad", "Eem"])
        # # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))

        # start_eHadron = index_by_objects["EFlowNeutralHadron"] 
        # n_eHadrons = fill_object(entry, start_eHadron, "EFlowNeutralHadron", "ET", 0, ["Ehad", "Eem"])


        # mu['MuIso'] = Iso(mu, muPt, muEta, muPhi) -1.
        # mu['EleIso'] = Iso(mu, elePt, eleEta, elePhi)
        # mu['ChHadIso'] = Iso(mu, trkPt, trkEta, trkPhi)
        # mu['NeuHadIso'] = Iso(mu, neuPt, neuEta, neuPhi)
        # mu['GammaIso'] = Iso(mu, gammaPt, gammaEta, gammaPhi)

        # fillIso("MuonTight", 'MuIso',  start_muons,Iso(muEta, muPhi, muPt, muEta, muPhi, muPt) -1.0)
        # fillIso("MuonTight", 'EleIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt) )
        # fillIso("MuonTight", 'ChHadIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt) )
        # fillIso("MuonTight", 'NeuHadIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt))
        # fillIso("MuonTight", 'GammaIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt))

        # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))
        

        

        # print("ETA",trkEta)
        
        
        #for i in range(n_values):
            
            #myMu = ROOT.TLorentzVector()
            #myMu.SetPtEtaPhiM(d["PT"][0].GetValue(entry), d["Eta"][0].GetValue(entry), d["Phi"][0].GetValue(entry), 0.)
            #myMuons.append({'PT':myMu.Pt(), 'Eta': myMu.Eta(), 'Phi': myMu.Phi(), 'Px': myMu.Px(), 'Py': myMu.Py(), 'Pz': myMu.Pz(), \
            #                    'X': 0., 'Y': 0., 'Z': 0., \
            #                   'Dxy': 0., 'Charge': d["Charge"][0].GetValue(entry), 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            #muEta[i] = (myMu.Eta()) 
            #muPhi[i] = (myMu.Phi())
            #muPt[i] = (myMu.Pt())
            #i = i +1 
        
        # # Look for Electrons
        # myElectrons = []
        # nEle = len(evt.Electron)
        # eleEta = np.zeros((nEle, 1))
        # elePhi = np.zeros((nEle, 1))
        # elePt = np.zeros((nEle, 1))
        # i = 0
        # for ele in evt.Electron:
        #     myEle = ROOT.TLorentzVector()
        #     myEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
        #     myElectrons.append({'PT': myEle.Pt(), 'Eta': myEle.Eta(), 'Phi': myEle.Phi(), 'Px': myEle.Px(), 'Py': myEle.Py(), 'Pz': myEle.P(), \
        #                         'X': 0., 'Y': 0., 'Z': 0., \
        #                         'Dxy': 0., 'Charge': ele.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     eleEta[i] = (myEle.Eta()) 
        #     elePhi[i] = (myEle.Phi())
        #     elePt[i] = (myEle.Pt())
        #     i =i +1

        # loop over tracks
        #for i in range(EFlowTrack_size):
        #    print EFlowTrack.PT[i]

        # Tracks (excluding electrons and muons)
        # nTracks = len(evt.EFlowTrack)
        # trkPt = np.zeros((nTracks, 1))
        # trkEta = np.zeros((nTracks, 1))
        # trkPhi = np.zeros((nTracks, 1))
        # myChargedHadrons = []
        # myMuAsTrk = []
        # myEleAsTrk = []
        # i = 0
        # for trk in evt.EFlowTrack:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(trk.PT, trk.Eta, trk.Phi, 0.)
        #     myTrk = {'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                  'X': trk.X, 'Y': trk.Y, 'Z': trk.Z, \
        #                  'Dxy': trk.Dxy, 'Charge': trk.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.}
        #     # ignore tracks identified as muons (through angular matching)
        #     if np.amin(DeltaRsq(myTrk, eleEta, elePhi)) < 1.E-3: 
        #         myEleAsTrk.append(myTrk)
        #         continue
        #     if np.amin(DeltaRsq(myTrk, muEta, muPhi)) < 1.E-3: 
        #         myMuAsTrk.append(myTrk)
        #         continue
        #     myChargedHadrons.append(myTrk)
        #     # for isolation
        #     trkEta[i] = trk.PT
        #     trkPhi[i] = trk.Eta
        #     trkPt[i] = trk.Phi
        #     i = i +1

        # # for each electron, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        # for ele in myElectrons:
        #     idx = Closest(ele, myEleAsTrk)
        #     if idx < 0: continue
        #     ele['Dxy'] = myEleAsTrk[idx]['Dxy']
        #     ele['X'] = myEleAsTrk[idx]['X']
        #     ele['Y'] = myEleAsTrk[idx]['Y']
        #     ele['Z'] = myEleAsTrk[idx]['Z']
        # # for each muon, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        # for mu in myMuons:
        #     idx = Closest(mu, myMuAsTrk)
        #     if idx < 0: continue
        #     mu['Dxy'] = myMuAsTrk[idx]['Dxy']
        #     mu['X'] = myMuAsTrk[idx]['X']
        #     mu['Y'] = myMuAsTrk[idx]['Y']
        #     mu['Z'] = myMuAsTrk[idx]['Z']
                
        # myPhotons = []
        # nGamma = len(evt.EFlowPhoton)
        # gammaEta = np.zeros((nGamma, 1))
        # gammaPhi = np.zeros((nGamma, 1))
        # gammaPt = np.zeros((nGamma, 1))
        # i = 0
        # for gamma in evt.EFlowPhoton:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(gamma.ET, gamma.Eta, gamma.Phi, 0.)
        #     myPhotons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                           'X': 0., 'Y': 0., 'Z': 0., \
        #                           'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     gammaEta[i] = p.Eta()
        #     gammaPhi[i] = p.Phi()
        #     gammaPt[i] = p.Pt()
        #     i = i +1

        # neutral hadrons
        # myNeutralHadrons = []
        # nNeuHad = len(evt.EFlowNeutralHadron)
        # neuEta = np.zeros((nNeuHad, 1))
        # neuPhi = np.zeros((nNeuHad, 1))
        # neuPt = np.zeros((nNeuHad, 1))
        # i = 0
        # for NeuHad in evt.EFlowNeutralHadron:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(NeuHad.ET, NeuHad.Eta, NeuHad.Phi, 0.)
        #     myNeutralHadrons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                                  'X': 0., 'Y': 0., 'Z': 0., \
        #                                  'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     neuEta[i] = NeuHad.Eta
        #     neuPhi[i] = NeuHad.Phi
        #     neuPt[i] = NeuHad.ET
        #     i = i +1

        # # compute isolation
        # for mu in myMuons:
        #     mu['MuIso'] = Iso(mu, muPt, muEta, muPhi) -1.
        #     mu['EleIso'] = Iso(mu, elePt, eleEta, elePhi)
        #     mu['ChHadIso'] = Iso(mu, trkPt, trkEta, trkPhi)
        #     mu['NeuHadIso'] = Iso(mu, neuPt, neuEta, neuPhi)
        #     mu['GammaIso'] = Iso(mu, gammaPt, gammaEta, gammaPhi)
            
        # for ele in myElectrons:
        #     ele['MuIso'] = Iso(ele, muPt, muEta, muPhi)
        #     ele['EleIso'] = Iso(ele, elePt, eleEta, elePhi) -1.
        #     ele['ChHadIso'] = Iso(ele, trkPt, trkEta, trkPhi)
        #     ele['NeuHadIso'] = Iso(ele, neuPt, neuEta, neuPhi)
        #     ele['GammaIso'] = Iso(ele, gammaPt, gammaEta, gammaPhi)

        # for gamma in myPhotons:
        #     gamma['EleIso'] = Iso(gamma, elePt, eleEta, elePhi) 
        #     gamma['MuIso'] = Iso(gamma, muPt, muEta, muPhi)
        #     gamma['ChHadIso'] = Iso(gamma, trkPt, trkEta, trkPhi)
        #     gamma['NeuHadIso'] = Iso(gamma, neuPt, neuEta, neuPhi)
        #     gamma['GammaIso'] = Iso(gamma, gammaPt, gammaEta, gammaPhi) -1

        # for p in myNeutralHadrons:
        #     p['EleIso'] = Iso(p, elePt, eleEta, elePhi)
        #     p['MuIso'] = Iso(p, muPt, muEta, muPhi)
        #     p['ChHadIso'] = Iso(p, trkPt, trkEta, trkPhi)
        #     p['NeuHadIso'] = Iso(p, neuPt, neuEta, neuPhi) -1
        #     p['GammaIso'] = Iso(p, gammaPt, gammaEta, gammaPhi) 

                   # print(pandas_out[obj][['Eta','Phi','X',  'Y',  'Z',  'Dxy'] ])

        # index_by_objects["Electron"] += n_electrons
        # index_by_objects["Photon"] += n_photons    
        # index_by_objects["MissingET"] += n_missingET
        # index_by_objects["EFlowTrack"] += n_tracks
        # index_by_objects["EFlowPhoton"] += n_ePhotons
        # index_by_objects["EFlowNeutralHadron"] += n_eHadrons

        # start_tracks = index_by_objects["EFlowTrack"] 
        # n_tracks = fill_object(entry, start_tracks, "EFlowTrack", "PT", 0, ["X","Y","Z","Dxy"])
        # trkEta, trkPhi, trkPt = getEtaPhiPTasNumpy("EFlowTrack", start_tracks, n_tracks)


        # start_muons = index_by_objects["MuonTight"] 
        # n_muons = fill_object(entry, start_muons, "MuonTight", "PT", 0, ["Charge"])
        # muEta, muPhi, muPt = getEtaPhiPTasNumpy("MuonTight", start_muons, n_muons)
        # matches = trackMatch(muEta, muPhi, trkEta, trkPhi)
        # fillTrackMatch("MuonTight", matches, start_muons, start_tracks)


        # start_electrons = index_by_objects["Electron"] 
        # n_electrons = fill_object(entry, start_electrons, "Electron", "PT", 0, ["Charge"])
        # eleEta, elePhi, elePt = getEtaPhiPTasNumpy("Electron", start, n_electrons)
        # matches = trackMatch(eleEta, elePhi, trkEta, trkPhi)
        # fillTrackMatch("Electron", matches, start_electrons, start_tracks)

        # start_gamma = index_by_objects["Photon"] 
        # n_photons = fill_object(entry, start_gamma, "Photon", "PT", 0, [])
        # gammaEta, gammaPhi, gammaPt = getEtaPhiPTasNumpy("Photon", start_gamma, n_photons)
       

        # print("ETA",gammaEta)
        # print("ETA",muEta)
        # print("ETA",eleEta)

        # start_missingET = index_by_objects["MissingET"] 
        # n_missingET = fill_object(entry, start_missingET, "MissingET", "MET", 0, [])
        # # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))

        # start_ePhoton = index_by_objects["EFlowPhoton"] 
        # n_ePhotons = fill_object(entry, start_ePhoton, "EFlowPhoton", "ET", 0, ["Ehad", "Eem"])
        # # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))

        # start_eHadron = index_by_objects["EFlowNeutralHadron"] 
        # n_eHadrons = fill_object(entry, start_eHadron, "EFlowNeutralHadron", "ET", 0, ["Ehad", "Eem"])


        # mu['MuIso'] = Iso(mu, muPt, muEta, muPhi) -1.
        # mu['EleIso'] = Iso(mu, elePt, eleEta, elePhi)
        # mu['ChHadIso'] = Iso(mu, trkPt, trkEta, trkPhi)
        # mu['NeuHadIso'] = Iso(mu, neuPt, neuEta, neuPhi)
        # mu['GammaIso'] = Iso(mu, gammaPt, gammaEta, gammaPhi)

        # fillIso("MuonTight", 'MuIso',  start_muons,Iso(muEta, muPhi, muPt, muEta, muPhi, muPt) -1.0)
        # fillIso("MuonTight", 'EleIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt) )
        # fillIso("MuonTight", 'ChHadIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt) )
        # fillIso("MuonTight", 'NeuHadIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt))
        # fillIso("MuonTight", 'GammaIso',  start_muons,Iso(muEta, muPhi, muPt, trkEta, trkPhi, trkPt))

        # gammaEta = np.array(islice(dicts_by_object["MissingET"]["Eta"], start, start+n_values))
        # gammaPhi = np.array(islice(dicts_by_object["MissingET"]["Phi"], start, start+n_values))
        # gammaPt = np.array(islice(dicts_by_object["MissingET"]["PT_ET"], start, start+n_values))
        

        

        # print("ETA",trkEta)
        
        
        #for i in range(n_values):
            
            #myMu = ROOT.TLorentzVector()
            #myMu.SetPtEtaPhiM(d["PT"][0].GetValue(entry), d["Eta"][0].GetValue(entry), d["Phi"][0].GetValue(entry), 0.)
            #myMuons.append({'PT':myMu.Pt(), 'Eta': myMu.Eta(), 'Phi': myMu.Phi(), 'Px': myMu.Px(), 'Py': myMu.Py(), 'Pz': myMu.Pz(), \
            #                    'X': 0., 'Y': 0., 'Z': 0., \
            #                   'Dxy': 0., 'Charge': d["Charge"][0].GetValue(entry), 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
            #muEta[i] = (myMu.Eta()) 
            #muPhi[i] = (myMu.Phi())
            #muPt[i] = (myMu.Pt())
            #i = i +1 
        
        # # Look for Electrons
        # myElectrons = []
        # nEle = len(evt.Electron)
        # eleEta = np.zeros((nEle, 1))
        # elePhi = np.zeros((nEle, 1))
        # elePt = np.zeros((nEle, 1))
        # i = 0
        # for ele in evt.Electron:
        #     myEle = ROOT.TLorentzVector()
        #     myEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
        #     myElectrons.append({'PT': myEle.Pt(), 'Eta': myEle.Eta(), 'Phi': myEle.Phi(), 'Px': myEle.Px(), 'Py': myEle.Py(), 'Pz': myEle.P(), \
        #                         'X': 0., 'Y': 0., 'Z': 0., \
        #                         'Dxy': 0., 'Charge': ele.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     eleEta[i] = (myEle.Eta()) 
        #     elePhi[i] = (myEle.Phi())
        #     elePt[i] = (myEle.Pt())
        #     i =i +1

        # loop over tracks
        #for i in range(EFlowTrack_size):
        #    print EFlowTrack.PT[i]

        # Tracks (excluding electrons and muons)
        # nTracks = len(evt.EFlowTrack)
        # trkPt = np.zeros((nTracks, 1))
        # trkEta = np.zeros((nTracks, 1))
        # trkPhi = np.zeros((nTracks, 1))
        # myChargedHadrons = []
        # myMuAsTrk = []
        # myEleAsTrk = []
        # i = 0
        # for trk in evt.EFlowTrack:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(trk.PT, trk.Eta, trk.Phi, 0.)
        #     myTrk = {'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                  'X': trk.X, 'Y': trk.Y, 'Z': trk.Z, \
        #                  'Dxy': trk.Dxy, 'Charge': trk.Charge, 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.}
        #     # ignore tracks identified as muons (through angular matching)
        #     if np.amin(DeltaRsq(myTrk, eleEta, elePhi)) < 1.E-3: 
        #         myEleAsTrk.append(myTrk)
        #         continue
        #     if np.amin(DeltaRsq(myTrk, muEta, muPhi)) < 1.E-3: 
        #         myMuAsTrk.append(myTrk)
        #         continue
        #     myChargedHadrons.append(myTrk)
        #     # for isolation
        #     trkEta[i] = trk.PT
        #     trkPhi[i] = trk.Eta
        #     trkPt[i] = trk.Phi
        #     i = i +1

        # # for each electron, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        # for ele in myElectrons:
        #     idx = Closest(ele, myEleAsTrk)
        #     if idx < 0: continue
        #     ele['Dxy'] = myEleAsTrk[idx]['Dxy']
        #     ele['X'] = myEleAsTrk[idx]['X']
        #     ele['Y'] = myEleAsTrk[idx]['Y']
        #     ele['Z'] = myEleAsTrk[idx]['Z']
        # # for each muon, find its view as EFlowTrack and retreive X, Y, Z, and Dxy
        # for mu in myMuons:
        #     idx = Closest(mu, myMuAsTrk)
        #     if idx < 0: continue
        #     mu['Dxy'] = myMuAsTrk[idx]['Dxy']
        #     mu['X'] = myMuAsTrk[idx]['X']
        #     mu['Y'] = myMuAsTrk[idx]['Y']
        #     mu['Z'] = myMuAsTrk[idx]['Z']
                
        # myPhotons = []
        # nGamma = len(evt.EFlowPhoton)
        # gammaEta = np.zeros((nGamma, 1))
        # gammaPhi = np.zeros((nGamma, 1))
        # gammaPt = np.zeros((nGamma, 1))
        # i = 0
        # for gamma in evt.EFlowPhoton:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(gamma.ET, gamma.Eta, gamma.Phi, 0.)
        #     myPhotons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                           'X': 0., 'Y': 0., 'Z': 0., \
        #                           'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     gammaEta[i] = p.Eta()
        #     gammaPhi[i] = p.Phi()
        #     gammaPt[i] = p.Pt()
        #     i = i +1

        # neutral hadrons
        # myNeutralHadrons = []
        # nNeuHad = len(evt.EFlowNeutralHadron)
        # neuEta = np.zeros((nNeuHad, 1))
        # neuPhi = np.zeros((nNeuHad, 1))
        # neuPt = np.zeros((nNeuHad, 1))
        # i = 0
        # for NeuHad in evt.EFlowNeutralHadron:
        #     p = ROOT.TLorentzVector()
        #     p.SetPtEtaPhiM(NeuHad.ET, NeuHad.Eta, NeuHad.Phi, 0.)
        #     myNeutralHadrons.append({'PT': p.Pt(), 'Eta': p.Eta(), 'Phi': p.Phi(), 'Px': p.Px(), 'Py': p.Py(), 'Pz': p.P(), \
        #                                  'X': 0., 'Y': 0., 'Z': 0., \
        #                                  'Dxy': 0., 'Charge': 0., 'ChHadIso': 0., 'NeuHadIso': 0., 'GammaIso': 0., 'MuIso': 0., 'EleIso': 0.})
        #     neuEta[i] = NeuHad.Eta
        #     neuPhi[i] = NeuHad.Phi
        #     neuPt[i] = NeuHad.ET
        #     i = i +1

        # # compute isolation
        # for mu in myMuons:
        #     mu['MuIso'] = Iso(mu, muPt, muEta, muPhi) -1.
        #     mu['EleIso'] = Iso(mu, elePt, eleEta, elePhi)
        #     mu['ChHadIso'] = Iso(mu, trkPt, trkEta, trkPhi)
        #     mu['NeuHadIso'] = Iso(mu, neuPt, neuEta, neuPhi)
        #     mu['GammaIso'] = Iso(mu, gammaPt, gammaEta, gammaPhi)
            
        # for ele in myElectrons:
        #     ele['MuIso'] = Iso(ele, muPt, muEta, muPhi)
        #     ele['EleIso'] = Iso(ele, elePt, eleEta, elePhi) -1.
        #     ele['ChHadIso'] = Iso(ele, trkPt, trkEta, trkPhi)
        #     ele['NeuHadIso'] = Iso(ele, neuPt, neuEta, neuPhi)
        #     ele['GammaIso'] = Iso(ele, gammaPt, gammaEta, gammaPhi)

        # for gamma in myPhotons:
        #     gamma['EleIso'] = Iso(gamma, elePt, eleEta, elePhi) 
        #     gamma['MuIso'] = Iso(gamma, muPt, muEta, muPhi)
        #     gamma['ChHadIso'] = Iso(gamma, trkPt, trkEta, trkPhi)
        #     gamma['NeuHadIso'] = Iso(gamma, neuPt, neuEta, neuPhi)
        #     gamma['GammaIso'] = Iso(gamma, gammaPt, gammaEta, gammaPhi) -1

        # for p in myNeutralHadrons:
        #     p['EleIso'] = Iso(p, elePt, eleEta, elePhi)
        #     p['MuIso'] = Iso(p, muPt, muEta, muPhi)
        #     p['ChHadIso'] = Iso(p, trkPt, trkEta, trkPhi)
        #     p['NeuHadIso'] = Iso(p, neuPt, neuEta, neuPhi) -1
        #     p['GammaIso'] = Iso(p, gammaPt, gammaEta, gammaPhi) 

        