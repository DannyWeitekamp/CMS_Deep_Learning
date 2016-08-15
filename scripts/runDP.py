import sys, os
scratch_path = "/scratch/daint/dweiteka/"
if(not scratch_path in sys.path):
    sys.path.append(scratch_path)

from CMS_SURF_2016.utils.archiving import DataProcedure

def main(archive_dir,hashcode):
    print(archive_dir,hashcode)
    dp = DataProcedure.find_by_hashcode(archive_dir=archive_dir,hashcode=hashcode,verbose=1)
    if(not dp.is_archived()):
    	dp.getData(archive=True)
    

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
