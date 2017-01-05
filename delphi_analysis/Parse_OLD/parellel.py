#WARNING THIS SCRIPT TAKES A LONG TIME TO RUN!
#Note Everythin is in natural units so C = 1
if __package__ is None:
    import sys, os
    sys.path.append(os.path.realpath("/data/shared/Software/"))
    sys.path.append(os.path.realpath("../../"))
import glob
import ntpath
from itertools import cycle, islice

import numpy as np
import pandas as pd

from CMS_Deep_Learning.old.data_parse import DataProcessingProcedure
from CMS_Deep_Learning.old.data_parse import ROOT_to_pandas
from CMS_Deep_Learning.old.data_parse import leaves_from_obj


#didit = False
def cullNonObservables(frame):
    #Status of 1 means that the particle is a stable product
    stable_cond = frame["Status"] == 1 
    #All even leptons are neutrinos which we can't measure
    notNeutrino_cond = frame["PID"] % 2 == 1
    parton_cond = np.abs(frame["PID"]) <= 8
    #Get all entries that satisfy the conditions
    frame = frame[stable_cond & notNeutrino_cond]
    #Drop the Status frame since we only needed it to see if the particle was stable
    frame = frame.drop(["Status"], axis=1)
    return frame

#Define the speed of light C
#Turns C=1 since everything is in natural units :/
#C = np.float64(2.99792458e8); #m/s
mass_of_electron = np.float64(0.0005109989461) #eV/c
mass_of_muon = np.float64(0.1056583715)      #eV/c
#def four_vec_func(inputs):
#        E = inputs[0]
#        Eta = inputs[1]
#        Phi = inputs[2]
#        PT = inputs[3]
#        E_over_c = E/C
#        px = E_over_c * np.sin(Phi) * np.cos(Eta) 
#        py = E_over_c * np.sin(Phi) * np.sin(Eta)
#        pz = E_over_c * np.cos(Phi)
#        print(np.sqrt(px*px + py*py), PT/C)
#        return [E_over_c, px, py, pz]

#def getPandasPhotons(filename):
#    four_vec_inputs, dummy = leaves_from_obj("Photon", ["PT", "Eta", "Phi"])
#    four_vec_proc = DataProcessingProcedure(lambda x: four_vec_from_PT(x,0), four_vec_inputs, ["E/c", "Px","Py","Pz"])

#    PID_proc = DataProcessingProcedure(lambda x:[22], [], ["PID"])
#    charge_proc = DataProcessingProcedure(lambda x:[0], [], ["Charge"])

#    columns=[four_vec_proc, PID_proc, charge_proc]
#    leaves, columns = leaves_from_obj("Photon", columns+["PT", "Eta", "Phi"])

#    photon_frame = ROOT_to_pandas(filename,
#                          leaves,
#                          columns=columns,
#                          verbosity=1)
#    return photon_frame


extra_data = ["PT", "Phi", "Eta"]


def four_vec_from_PT(inputs, M):
    #print(type(inputs[0]))
    PT = inputs[0] #Units ?
    Eta = inputs[1]
    Phi = inputs[2]
    #M has units of eV/(c^2)
    if(M == None):
        M = inputs[3]
    momentum_mag = PT * np.cosh(Eta)
    #if(~didit):
    #print(momentum_mag, PT, np.cosh(Eta))
    #    didit = True
    E_over_c = np.sqrt(np.square(M) + np.square(momentum_mag))
    #print(PT/np.sin(Phi))
    px = PT * np.cos(Phi) 
    py = PT * np.sin(Phi) 
    pz = PT * np.sinh(Eta)
    return [E_over_c, px, py, pz]

def getPandasParticles(filename, cull=True):
    #C=1 in natural units so no processing needs to be done
    E_over_c_proc = DataProcessingProcedure(lambda x:x[0], ["Particle.E"], ["E/c"])
    columns= [E_over_c_proc, "Px", "Py", "Pz", "PID", "Charge", "Status"]
    leaves, columns = leaves_from_obj("Particle", columns+extra_data)
    original_frame = ROOT_to_pandas(filename,
                                 leaves,
                                  columns=columns,
                                  verbosity=1)
    if(cull):
        particle_frame = cullNonObservables(original_frame)
    else:
        particle_frame = original_frame
    return particle_frame


def getPandasSpecificParticles(filename, name, mass=None, pid=None, chrg_def=-1, charge=None):
    if(mass != None):
        four_vec_inputs, dummy = leaves_from_obj(name, ["PT", "Eta", "Phi"])
    else:
        four_vec_inputs, dummy = leaves_from_obj(name, ["PT", "Eta", "Phi", "Mass"])
        
    four_vec_proc = DataProcessingProcedure(lambda x: four_vec_from_PT(x,mass)
                                            , four_vec_inputs, ["E/c", "Px","Py","Pz"])
    
    status_proc = DataProcessingProcedure(lambda x:[1], [], ["Status"])
    
    if(charge == None):
        if(pid != None):
            PID_charge_proc = DataProcessingProcedure(lambda x: [pid*chrg_def*x[0], x[0]]
                                                      , [name + ".Charge"], ["PID", "Charge"])
            columns=[four_vec_proc, PID_charge_proc, status_proc]
        else:
            columns=[four_vec_proc, "PID", "Charge", status_proc]
    else:
        charge_proc = DataProcessingProcedure(lambda x: [charge], [], ["Charge"])
        if(pid != None):
            PID_proc = DataProcessingProcedure(lambda x: [pid], [], ["PID"])
            columns=[four_vec_proc, PID_proc, charge_proc,status_proc]
        else:
            columns=[four_vec_proc, "PID", charge_proc,status_proc]
        
        
        
        
    
    leaves, columns = leaves_from_obj(name, columns+extra_data)

    #Extract the tables from the root file
    frame = ROOT_to_pandas(filename,
                          leaves,
                          columns=columns,
                          verbosity=1)
    return frame

def getPandasPhotons(filename):
    return getPandasSpecificParticles(filename, "Photon", mass=0, pid=22, charge=0)

def getPandasElectrons(filename):
    return getPandasSpecificParticles(filename, "Electron", mass=mass_of_electron, pid=11)

def getPandasTightMuons(filename):
    return getPandasSpecificParticles(filename, "MuonTight", mass=mass_of_muon, pid=13)

def getPandasJets(filename):
    return getPandasSpecificParticles(filename, "Jet", pid=100, chrg_def=1)

def getPandasMissingET(filename, name="MissingET"):  
    pid = 83
    if(name == "PuppiMissingET"): pid=84
    four_vec_inputs, dummy = leaves_from_obj(name, ["MET", "Eta", "Phi"])
    four_vec_proc = DataProcessingProcedure(lambda x: four_vec_from_PT(x,0)
                                            , four_vec_inputs, ["E/c", "Px","Py","Pz"])
    status_proc = DataProcessingProcedure(lambda x:[1], [], ["Status"])
    charge_proc = DataProcessingProcedure(lambda x: [0], [], ["Charge"])
    PID_proc = DataProcessingProcedure(lambda x: [pid], [], ["PID"])
    columns=[four_vec_proc, PID_proc, charge_proc, status_proc]
    met_proc = DataProcessingProcedure(lambda x: [x[0]], ["MissingET.MET"], ["PT"] )
    
    ex = [x if (x != "PT") else met_proc for x in extra_data]
    leaves, columns = leaves_from_obj(name, columns+ex)

    #Extract the tables from the root file
    frame = ROOT_to_pandas(filename,
                          leaves,
                          columns=columns,
                          verbosity=1)
    return frame
        
        


def getPandasAll(filename, cull=False, includePuppi=True):
    lst = [getPandasPhotons(filename),
                    getPandasElectrons(filename),
                    getPandasTightMuons(filename),
                    getPandasJets(filename),
                    getPandasParticles(filename, cull=cull),
                    getPandasMissingET(filename)]
    
    if(includePuppi):
        lst = lst + [getPandasMissingET(filename, "PuppiMissingET")]
    #Merge all these frames
    out = pd.concat(lst)
    return out

#def getPandasAll(filename):
#    out = pd.concat([getPandasPhotons(filename),getPandasParticles(filename, cull=False)])
#    return out

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

    
def storeAllUnjoined(filepath, outputDir, rerun=False):
    
    filename = os.path.splitext(ntpath.basename(filepath))[0]
    
    name_cols = [("EFlowTrack", ["PT", "Eta", "Phi", "Dxy", "Charge"]),
                 ("EFlowPhoton", ["ET", "Eta", "Phi", "Eem", "Ehad"]),
                 ("EFlowNeutralHadron", ["ET", "Eta", "Phi", "Eem", "Ehad"]),
                 ("Electron", ["PT", "Eta", "Phi", "Charge"]),
                 ("MuonTight", ["PT", "Eta", "Phi", "Charge"]),
                 ("MissingET", ["MET", "Eta", "Phi"])]
    for nc in name_cols:
        name = nc[0]
        cols = nc[1]
        out_file = outputDir + name + "/" + filename + ".h5"
        leaves, columns = leaves_from_obj(name,cols)
        if not os.path.exists(outputDir+name):
            os.makedirs(outputDir+name)
        if(rerun or os.path.isfile(out_file) == False):
            #print(rerun, ~os.path.isfile(out_file), out_file)
            frame = ROOT_to_pandas(filepath, leaves, columns=columns)
            frame.to_hdf(out_file, 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("EFlowTrack", ["PT", "Eta", "Phi", "Dxy", "Charge"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "EFlowTrack/" + filename + ".h5", 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("EFlowPhoton", ["ET", "Eta", "Phi", "Eem", "Ehad"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "EFlowPhoton/" + filename + ".h5", 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("EFlowNeutralHadron", ["ET", "Eta", "Phi", "Eem", "Ehad"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "EFlowNeutralHadron/" + filename + ".h5", 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("Electron", ["PT", "Eta", "Phi", "Charge"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "Electron/" + filename + ".h5", 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("MuonTight", ["PT", "Eta", "Phi", "Charge"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "MuonTight/" + filename + ".h5", 'data', mode='w')
    
    # leaves, columns = leaves_from_obj("MissingET", ["MET", "Eta", "Phi"])
    # frame = ROOT_to_pandas(filepath, leaves, columns=columns)
    # frame.to_hdf(outputDir + "MissingET/" + filename + ".h5", 'data', mode='w')
    
    
def makeJobs(filename, job_types,
             directory="/data/shared/Delphes/",
             unjoined_folder="/pandas_unjoined/",
             joined_folder="/pandas_joined/"):
    files = glob.glob(directory + filename + "/*.root")
    unjoined_dir = directory + filename + unjoined_folder
    joined_dir = directory + filename + joined_folder
    if not os.path.exists(joined_dir):
        os.makedirs(joined_dir)
    
    jobs = []
    
    for f in files:
        f_name = os.path.splitext(ntpath.basename(f))[0]
        for j_type in job_types:
            if(j_type == "unjoined"):
                jobs.append((j_type,f, unjoined_dir))
            elif(j_type == "joined"):
                jobs.append((j_type,f, joined_dir + f_name + ".h5"))

    return jobs
    
    
#def groupEntriesToArrays(frame, select):
#    out = []
#    m = frame['Entry'].max()
#    for entry in range(0, m+1):
#        cond = frame['Entry'] == entry
#        entry_frame = frame[cond]
#        entry_frame = entry_frame[select]
#        arr = np.array(entry_frame)
#        np.random.shuffle(arr)
#        out.append(arr)
#    return out

#def groupEntriesToArrays(frame, select):
#    grouped = frame.groupby(["Entry"]).apply(lambda df: df[select].tolist())
    #print(grouped)
#    return grouped

# ttbar_files = glob.glob("/data/shared/Delphes/ttbar_lepFilter_13TeV/*.root")
# WJet_files = glob.glob("/data/shared/Delphes/wjets_lepFilter_13TeV/*.root")
# WJet_files = glob.glob("/data/shared/Delphes/qcd_lepFilter_13TeV/*.root")


# ttbar_unjoined_dir = "/data/shared/Delphes/ttbar_lepFilter_13TeV/pandas_unjoined/"
# WJet_unjoined_dir = "/data/shared/Delphes/wjets_lepFilter_13TeV/pandas_unjoined/"
# QCD_unjoined_dir = "/data/shared/Delphes/qcd_lepFilter_13TeV/pandas_unjoined/"
    
# for(_dir in [ttbar_unjoined_dir, WJet_unjoined_dir, QCD_unjoined_dir]):
#     if not os.path.exists(_dir):
#         os.makedirs(_dir)
    
# ttbar_jobs_unjoined = []
# WJet_jobs_unjoined = []
# QCD_jobs_unjoined = []

# ttbar_jobs_joined = []
# WJet_jobs_joined = []
# QCD_jobs_joined = []
    
# for ttbar_file in ttbar_files:
#     ttbar_filename = os.path.splitext(ntpath.basename(ttbar_file))[0]
#     ttbar_jobs.append((ttbar_file, ttbar_out_dir + ttbar_filename + ".h5"))

# for WJet_file in WJet_files:
#     WJet_filename = os.path.splitext(ntpath.basename(WJet_file))[0]
#     WJet_jobs.append((WJet_file, WJet_out_dir + WJet_filename + ".h5"))
    
# for WJet_file in WJet_files:
#     WJet_filename = os.path.splitext(ntpath.basename(WJet_file))[0]
#     WJet_jobs.append((WJet_file, WJet_out_dir + WJet_filename + ".h5"))
    
#TODO: Add QCD
ttbar_jobs = makeJobs("ttbar_lepFilter_13TeV", ["unjoined"])
WJet_jobs = makeJobs("wjets_lepFilter_13TeV", ["unjoined"])
qcd_jobs = makeJobs("qcd_lepFilter_13TeV", ["unjoined"])
    
jobs = roundrobin(ttbar_jobs, WJet_jobs, qcd_jobs)


from multiprocessing.dummy import Pool
import time

#queue = Queue()

#for job in jobs:
#    queue.put(job)

#def getAndDoJob(queue):
#    job = queue.get()
#    print(job[2])

job_itr = iter(jobs)

def doJob(job):
    time.sleep(int(1))
    if(job[0] == "unjoined"):
        #print()
        #print(job[0])
        print("Starting Job:", job[1])
        #print(job[2])
        storeAllUnjoined(job[1], job[2])
    elif(job[0] == "joined"):
        print(job)
    return job[1]

pool = Pool(4)

def mycallback(x):
    print("Finished Job:", job[1])
    sys.stdout.flush()

results = []
for job in jobs:
    r = pool.apply_async(doJob, args=[job], callback=mycallback)
    results.append(r)
#pool.close()
#pool.join() 
for r in results:
    r.wait()
#processes = []
#for i in range(4):
#    p = Process(target=getAndDoJob, args=(queue,))
#    p.daemon = True
#    p.start()
#    processes.append(p)
    #p.join()
#for p in processes:
#    p.join()
#pool = ThreadPool(4)


#pool.map_async(doJob, jobs, callback=mycallback)


#for job in jobs:
    
        #frame = getPandasAll(job[1])
        #frame.to_hdf(job[2], 'data', mode='w')
#print(job[0])
#print(job[1])
#print(job[2])
#storeAllUnjoined(job[1], job[2])
