#WARNING THIS SCRIPT TAKES A LONG TIME TO RUN!
#Note Everythin is in natural units so C = 1
import sys, os
if __package__ is None:
    import sys, os
    sys.path.append(os.path.realpath("/data/shared/Software/"))
    sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.data_parse import *
#from CMS_SURF_2016.utils.data_parse import leaves_from_obj
import ROOT
from ROOT import TTree
import numpy as np
import pandas as pd
import ntpath
import glob
from itertools import cycle, islice
import time
import getopt



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


def storeAllUnjoined(filepath, outputdir, rerun=False):
   lst = [        ("NumValues", "getPandasNumValues(filepath)"),
                  ("Photon", "getPandasPhotons(filepath)"),
                  ("Electron", "getPandasElectrons(filepath)"),
                  ("MuonTight", "getPandasTightMuons(filepath)"),
                  ("MissingET", "getPandasMissingET(filepath)"),
                  ("EFlowPhoton", "getPandasEFlowParticle(filepath, name='EFlowNeutralHadron')"),
                  ("EFlowNeutralHadron", "getPandasEFlowParticle(filepath, name='EFlowPhoton')"),
                  ("EFlowTrack", "getPandasEFlowTrack(filepath)")]
    
   filename = os.path.splitext(ntpath.basename(filepath))[0]
   out_file = outputdir + filename + ".h5"
   print(out_file)
   store = pd.HDFStore(out_file)
   keys = store.keys()
   print("KEYS:", keys)
   for tup in lst:
      if(rerun or (("/"+tup[0]) in keys) == False):
         #print(rerun, ~os.path.isfile(out_file), out_file)
         frame = eval(tup[1])
         print(frame)
         store.put(tup[0], frame, format='table')
   store.close()
            
        

def storeAllJoined(filepath, outputfile, rerun=False):
   if(rerun or os.path.isfile(outputfile) == False):
      #print(rerun, ~os.path.isfile(out_file), out_file)
      frame = getPandasAll(filepath)
      frame.to_hdf(outputfile, 'data', mode='w', data_columns=True)
    

    
def makeJobs(filename, job_types,
             directory="/data/shared/Delphes/",
             unjoined_folder="/pandas_unjoined/",
             joined_folder="/pandas_joined/"):
   files = glob.glob(directory + filename + "/*.root")
   unjoined_dir = directory + filename + unjoined_folder
   joined_dir = directory + filename + joined_folder
   if not os.path.exists(joined_dir):
      os.makedirs(joined_dir)
   if not os.path.exists(unjoined_dir):
      os.makedirs(unjoined_dir)
    
   jobs = []
    
   for f in files:
      f_name = os.path.splitext(ntpath.basename(f))[0]
      for j_type in job_types:
         if(j_type == "unjoined"):
            jobs.append((j_type,f, unjoined_dir))
         elif(j_type == "joined"):
            jobs.append((j_type,f, joined_dir + f_name + ".h5"))

    return jobs

def doJob(job):
   if(job[0] == "unjoined"):
      #print("Started: ", job[1])
      storeAllUnjoined(job[1], job[2])
   elif(job[0] == "joined"):
      #print(job)
      storeAllJoined(job[1], job[2])
   return job[1]


def main(data_dir, argv):
  
   joined = False
   screwup_error = 'delphes_to_pandas.py <inputdir> -j'
   try:
      opts, args = getopt.getopt(argv,'j')
   except getopt.GetoptError:
      print screwup_error
      sys.exit(2)
  
   for opt, arg in opts:
      # print(opt, arg)
      if opt in ("-j", "--joined"):
         joined = True
   
   t = "unjoined" if joined  == False else "joined"
   print(t)
   jobs = makeJobs( data_dir, [t])
   for job in jobs:
      #print(job)
      doJob(job)




if __name__ == "__main__":
   main(sys.argv[1],sys.argv[2:])
