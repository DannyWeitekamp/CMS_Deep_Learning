import sys
import time

repo_outerdir = "/scratch/daint/dweiteka/"
if(not repo_outerdir in sys.path):
    sys.path.append(repo_outerdir)
#%matplotlib inline
# import sys, os
# if __name__ == "__main__":
#     username = "dweiteka"
#     if(len(sys.argv) > 1):
#         username = sys.argv[1]


import socket
if("daint" in socket.gethostname()):
    DELPHES_DIR = "/scratch/daint/" + "dweiteka" +  "/Delphes/"
    SOFTWAR_DIR = "/scratch/daint/" + "dweiteka" + "/"
    STORE_TYPE = "msg"
else:
    DELPHES_DIR = "/data/shared/Delphes/"
    SOFTWAR_DIR = "/data/shared/Software/"
    STORE_TYPE = "h5"

archive_dir = DELPHES_DIR+"CSCS_output/keras_archive/"

imports_ok = False
start_time = time.clock()
while(time.clock() - start_time < 5):
    try:
        from CMS_Deep_Learning.storage.archiving import KerasTrial, DataProcedure,get_all_trials
        from CMS_Deep_Learning.postprocessing.metrics import *
        from CMS_Deep_Learning.layers.lorentz import Lorentz
        from CMS_Deep_Learning.layers.slice import Slice
        from CMS_Deep_Learning.storage.rsyncUtils import rsyncStorable
        imports_ok = True
        break
    except Exception as e:
        print(e)
        print("Failed import trying again...")
        sys.stdout.flush()
        time.sleep(1)
        continue

if(not imports_ok):
    raise IOError("Failed to import CMS_Deep_Learning or keras, ~/.keras/keras.json is probably being read by multiple processes")

print("WE Got Imports")
#def main():
trials = get_all_trials(archive_dir)
print("Got %r Trials" % len(trials))
for trial in trials:
    print("START %r" % trial.hash())
    print(getTrialError(trial,custom_objects={"Lorentz":Lorentz,"Slice": Slice}, ignoreAssert=True))
    #rsyncStorable(trial.hash(), archive_dir, "dweitekamp@titans.hep.caltech.edu:/data/shared/Delphes/CSCS_output/keras_archive")

# if __name__ == "__main__":
    # main()

