import sys,os
import re
import socket
from CMS_SURF_2016.utils.archiving import DataProcedure, KerasTrial
from CMS_SURF_2016.layers.lorentz import Lorentz
from CMS_SURF_2016.layers.slice import Slice

def batchAssertArchived(dps, time_str="01:00:00",repo="/scratch/daint/dweiteka/CMS_SURF_2016/", dp_out_dir='/scratch/daint/dweiteka/dp_out/'):
    scripts_dir = repo + "scripts/"

    unarchived = []
    dependencies = []
    for dp in dps:
        if(not dp.is_archived()):
            unarchived.append(dp)

    if("daint" in socket.gethostname()):
        if(not os.path.exists(scripts_dir + "tmp/")):
            os.makedirs(scripts_dir + "tmp/")
        runDPs_file = scripts_dir + "tmp/runDPs.sh"
        f = open(runDPs_file, 'w')
        os.chmod(runDPs_file, 0o777)
        f.write("#!/bin/bash\n")
        for u in unarchived:
            u.write()
            ofile = dp_out_dir + u.hash()[:5] + ".%j"
            print("OutFile: ",ofile)
            f.write('sbatch -t %s -o %s -e %s %srunDP.sh %s %s %s\n' % (time_str,ofile,ofile,scripts_dir, repo,u.archive_dir,u.hash()))
            
        f.close()
        
        out = os.popen(scripts_dir+"tmp/runDPs.sh").read()
        print("THIS IS THE OUTPUT:",out)
        dep_clause = ""
        matches = re.findall("Submitted batch job [0-9]+", out) 

        dependencies = [re.findall("[0-9]+", m)[0] for m in matches]
        if(len(dependencies) > 0):
            dep_clause = "--dependency=afterok:" + ":".join(dependencies)
    else:
        for u in unarchived:
            u.getData(archive=True)
    return dependencies

def batchExecuteAndTestTrials(tups, time_str="12:00:00", repo="/scratch/daint/dweiteka/CMS_SURF_2016/", trial_out_dir='/scratch/daint/dweiteka/trial_out/'):
    isdaint = "daint" in socket.gethostname()
    scripts_dir = repo + "scripts/" 
    for trial, test, num_test, deps in tups:
        archive_dir = trial.archive_dir
        hashcode = trial.hash()
        test.write()
        test_hashcode = test.hash()
        if(isdaint):
            dep_clause = "" if len(deps)==0 else "--dependency=afterok:" + ":".join(deps)
            ofile = trial_out_dir + hashcode[:5] + ".%j"
            sbatch = 'sbatch -t %s -o %s -e %s %s ' % (time_str,ofile,ofile,dep_clause)
            sbatch += '%srunTrial.sh %s %s %s %s %s\n' % (scripts_dir,repo,archive_dir,hashcode, test_hashcode, num_test)
            print(sbatch)
            out = os.popen(sbatch).read()
            print("THIS IS THE OUTPUT:",out)
        else:
            trial = KerasTrial.find_by_hashcode(archive_dir, hashcode)
            trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice})

            test = DataProcedure.find_by_hashcode(archive_dir,test_hashcode)
            print("T@:", type(test))
            trial.test(test_proc=test,
                         test_samples=num_test,
                         custom_objects={"Lorentz":Lorentz,"Slice": Slice})