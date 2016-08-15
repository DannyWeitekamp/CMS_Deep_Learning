import sys, os
scratch_path = "/scratch/daint/dweiteka/"
if(not scratch_path in sys.path):
    sys.path.append(scratch_path)

from CMS_SURF_2016.utils.archiving import KerasTrial, DataProcedure
from CMS_SURF_2016.layers.lorentz import Lorentz
from CMS_SURF_2016.layers.slice import Slice

def main(archive_dir,hashcode, test_hashcode, num_test):
    print("STARTING: %s" % hashcode)
    sys.stdout.flush()
    trial = KerasTrial.find_by_hashcode(archive_dir, hashcode)
    print("EXECUTING: %s" % hashcode)
    sys.stdout.flush() 
    trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice})
    
    raise NotImplementedError("Will not run test, evaluate_generator acts weird on CSCS")    
    print("TESTING: %s, num_samples: %r" % (hashcode,num_test))
    sys.stdout.flush()
    test = DataProcedure.find_by_hashcode(archive_dir,test_hashcode)
    metrics = trial.test(test_proc=test,
                 test_samples=num_test,
                 custom_objects={"Lorentz":Lorentz,"Slice": Slice})
    print("DONE: %r" % metrics)
    
    

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

