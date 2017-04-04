import os
import re
import socket
import sys
from multiprocessing import Process
from time import sleep

import numpy as np


from CMS_Deep_Learning.layers.lorentz import Lorentz
from CMS_Deep_Learning.layers.slice import Slice
from CMS_Deep_Learning.storage.archiving import DataProcedure, KerasTrial


def batchAssertArchived(dps, num_processes=1, time_str="01:00:00",repo="/scratch/snx3000/dweiteka/CMS_Deep_Learning/", dp_out_dir='/scratch/snx3000/dweiteka/dp_out/', verbose=1):
    '''Makes sure that a list of DataProcedures are archived before training starts. When used on Daint, runs each DataProcedure in different batches and outputs
        a list of job numbers corresponding each batch. These can be passed to batchExecuteAndTestTrials to make sure that the trials are run only after the
        DPs have completed their preprocessing and archived the result.
    '''
    scripts_dir = repo + "scripts/"

    unarchived = []
    dependencies = []
    for dp in dps:
        if(not dp.is_archived()):
            unarchived.append(dp)

    if("daint" in socket.gethostname()):
        if(not os.path.exists(scripts_dir + "tmp/")):
            os.makedirs(scripts_dir + "tmp/")
        if(not os.path.exists(dp_out_dir)):
            os.makedirs(dp_out_dir)
        runDPs_file = scripts_dir + "tmp/runDPs.sh"
        f = open(runDPs_file, 'w')
        os.chmod(runDPs_file, 0o777)
        f.write("#!/bin/bash\n")
        for u in unarchived:
            u.write()
            ofile = dp_out_dir + u.hash()[:5] + ".%j"
            if(verbose >= 1): print("OutFile: ",ofile)
            f.write('sbatch -C gpu -t %s -o %s -e %s %srunDP.sh %s %s %s\n' % (time_str,ofile,ofile,scripts_dir, repo,u.archive_dir,u.hash()))
            
        f.close()
        
        out = os.popen(scripts_dir+"tmp/runDPs.sh").read()
        if(verbose >= 1): print("THIS IS THE OUTPUT:",out)
        dep_clause = ""
        matches = re.findall("Submitted batch job [0-9]+", out) 

        dependencies = [re.findall("[0-9]+", m)[0] for m in matches]
        if(len(dependencies) > 0):
            dep_clause = "--dependency=afterok:" + ":".join(dependencies)
    else:
        if(len(unarchived) == 0):
            return None

        archive_dir = unarchived[0].archive_dir
        for u in unarchived:
            u.write()

        hashes = [u.hash() for u in unarchived]
        def f(hashes,archive_dir, verbose=0,i=0):
            from CMS_Deep_Learning.storage.archiving import DataProcedure
            if (verbose >= 1): print("Batch process %r started." % i)
            for h in hashes:
                u = DataProcedure.find(archive_dir=archive_dir, hashcode=h)
                u.get_data(archive=True, verbose=verbose)
                if(verbose >= 1): print("From process %r." % i)

        processes = []
        if(verbose >= 1): print("Starting batchAssertArchived starting with %r/%r DataProcedures" % (len(dps)-len(unarchived), len(dps)))
        splits = np.array_split(hashes, num_processes)
        for i, sublist in enumerate(splits[1:]):
            print("Thread %r Started" % i)
            p = Process(target=f, args=(sublist,archive_dir,verbose,i+1))
            processes.append(p)
            p.start()
            sleep(.001)
        try:
            f(splits[0],archive_dir,verbose=verbose)
        except:
            for p in processes:
                p.terminate()
        for p in processes:
            p.join()
        if False in [u.is_archived() for u in unarchived]:
            print("Batch Assert Failed")
            pass#batchAssertArchived(dps, num_processes=num_processes)

        if(verbose >= 1): sys.stdout.write("Done.")
    return dependencies

def batchExecuteAndTestTrials(tups, time_str="24:00:00", repo="/scratch/snx3000/dweiteka/CMS_Deep_Learning/", trial_out_dir='/scratch/snx3000/dweiteka/trial_out/',use_mpi=False, verbose=1):
    '''Takes in a list of tuples 'tups' of the form (trial (a KerasTrial), test (a DataProcedure), num_test (an Integer), deps (a list)), and executes/tests 
        each trial, either in in order or in separate batches in the case of CSCS.
    '''
    isdaint = "daint" in socket.gethostname()
    scripts_dir = repo + "scripts/" 
    for trial, test, num_test, deps in tups:
        archive_dir = trial.archive_dir
        hashcode = trial.hash()

        test_hashcode = None
        if(test != None):
            
            test.write()
            test_hashcode = test.hash()
        if(isdaint):
            if(not os.path.exists(trial_out_dir)):
                os.makedirs(trial_out_dir)
            dep_clause = "" if len(deps)==0 else "--dependency=afterok:" + ":".join(deps)
            ofile = trial_out_dir + hashcode[:5] + ".%j"
            sbatch = 'sbatch -C gpu -t %s -o %s -e %s %s ' % (time_str,ofile,ofile,dep_clause)
            sbatch += '%srunTrial.sh %s %s %s %s %s\n' % (scripts_dir,repo,archive_dir,hashcode, test_hashcode, num_test)
            if(verbose >=1): print(sbatch)
            out = os.popen(sbatch).read()
            if(verbose >=1): print("THIS IS THE OUTPUT:",out)
        else:
            if(use_mpi):
                trial = KerasTrial.find(archive_dir, hashcode) 
            else:
                from CMS_Deep_Learning.storage.MPIArchiving import MPI_KerasTrial
                trial = MPI_KerasTrial.find(archive_dir, hashcode)
            if(verbose >=1): print("EXECUTE %r" % trial.hash())
            trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice})

            if(test_hashcode != None):
                if(verbose >=1): print("TEST %r" % trial.hash())
                test = DataProcedure.find(archive_dir, test_hashcode)
                trial.test(test_proc=test,
                             test_samples=num_test,
                             custom_objects={"Lorentz":Lorentz,"Slice": Slice})