import sys, os
import time

repo_outerdir = sys.argv[1]+"../"
if(not repo_outerdir in sys.path):
    sys.path.append(repo_outerdir)

imports_ok = False
start_time = time.clock()
while(time.clock() - start_time < 5):
    try:
        from CMS_SURF_2016.utils.archiving import KerasTrial, DataProcedure
        from CMS_SURF_2016.layers.lorentz import Lorentz
        from CMS_SURF_2016.layers.slice import Slice
        from CMS_SURF_2016.utils.gitutils import addCommitPushDir
        imports_ok = True
        break
    except Exception as e:
        print("Failed import trying again...")
        sys.stdout.flush()
        time.sleep(1)
        continue

if(not imports_ok):
    raise IOError("Failed to import CMS_SURF_2016 or keras, check that ~/.keras/keras.json is not corrupted")


def main(archive_dir,hashcode, test_hashcode, num_test):
    print("STARTING: %s" % hashcode)
    sys.stdout.flush()
    trial = KerasTrial.find_by_hashcode(archive_dir, hashcode)
    print("EXECUTING: %s" % hashcode)
    sys.stdout.flush() 
    trial.execute(custom_objects={"Lorentz":Lorentz,"Slice": Slice})
    addCommitPushDir(trial.get_path())

    return
    raise NotImplementedError("Will not run test, evaluate_generator acts weird on CSCS")    
    print("TESTING: %s, num_samples: %r" % (hashcode,num_test))
    sys.stdout.flush()
    test = DataProcedure.find_by_hashcode(archive_dir,test_hashcode)
    metrics = trial.test(test_proc=test,
                 test_samples=num_test,
                 custom_objects={"Lorentz":Lorentz,"Slice": Slice})
    print("DONE: %r" % metrics)
    
    

if __name__ == "__main__":
    main(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

