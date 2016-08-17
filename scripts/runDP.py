import sys, os
import time

repo_outerdir = sys.argv[1]+"../"
if(not repo_outerdir in sys.path):
    sys.path.append(repo_outerdir)

imports_ok = False
start_time = time.clock()
while(time.clock() - start_time < 5):
    try:
        from CMS_SURF_2016.utils.archiving import DataProcedure
        break
    except Exception as e:
    	print("Failed import trying again...")
        time.sleep(1)
        continue

if(not imports_ok):
    raise IOError("Failed to import CMS_SURF_2016 or keras, check that ~/.keras/keras.json is not corrupted")


def main(archive_dir,hashcode):
    print(archive_dir,hashcode)
    dp = DataProcedure.find_by_hashcode(archive_dir=archive_dir,hashcode=hashcode,verbose=1)
    if(not dp.is_archived()):
    	dp.getData(archive=True)
    

if __name__ == "__main__":
    main(sys.argv[2],sys.argv[3])
