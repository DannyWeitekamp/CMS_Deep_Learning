import sys
import time

repo_outerdir = sys.argv[1]+"../"
if(not repo_outerdir in sys.path):
    sys.path.append(repo_outerdir)

imports_ok = False
start_time = time.clock()
while(time.clock() - start_time < 60):
    try:
        from CMS_Deep_Learning.storage.archiving import DataProcedure
        imports_ok = True
        break
    except Exception as e:
    	print("Failed import trying again...")
        time.sleep(1)
        continue

if(not imports_ok):
    raise IOError("Failed to import CMS_Deep_Learning or keras, ~/.keras/keras.json is probably being read by multiple processes")


def main(archive_dir,hashcode):
    print(archive_dir,hashcode)
    sys.stdout.flush()
    dp = DataProcedure.find(archive_dir=archive_dir, hashcode=hashcode, verbose=1)
    if(not dp.is_archived()):
    	dp.get_data(archive=True)
    

if __name__ == "__main__":
    main(sys.argv[2],sys.argv[3])
