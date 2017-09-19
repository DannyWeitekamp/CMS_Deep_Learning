import numpy as np
import glob,os,sys,argparse

if __package__ is None:
    #sys.path.append(os.path.realpath("../"))
    sys.path.append(os.path.realpath(__file__+"/../../../"))

from CMS_Deep_Learning.io import get_sizes_meta_dict,size_from_meta
from CMS_Deep_Learning.preprocessing.pandas_to_numpy import to_shuffled_numpy,PARTICLE_OBSERVS,HLF_OBSERVS,DEFAULT_RPE,_checkDir
import h5py


def gen_std_stats(sources):
        print("Computing mean & std from sample:")

        sources = [_checkDir(s) for s in sources]
        
        tots = []
        for s in sources:
            files = glob.glob(os.path.abspath(s) + "/*.h5")
            tot = 0
            sizesDict = get_sizes_meta_dict(s)
            for f in files:
                tot += size_from_meta(f, sizesDict=sizesDict)
            tots.append(tot)
            
        N_MANY_SAMPLES = min(tots)
        particles, _, hlf,_ = to_shuffled_numpy(sources, 0, N_MANY_SAMPLES)
        particles_flat = particles.reshape((len(sources)*N_MANY_SAMPLES*DEFAULT_RPE['Particles'], len(PARTICLE_OBSERVS)))
        hlf_flat = hlf.reshape((len(sources) * N_MANY_SAMPLES*DEFAULT_RPE['HLF'], len(HLF_OBSERVS)))
        particle_mean = np.mean(particles_flat,axis=0)
        particle_std = np.std(particles_flat,axis=0)
        hlf_mean = np.mean(hlf_flat,axis=0)
        hlf_std = np.std(hlf_flat,axis=0)
        
        return  particle_mean,particle_std,hlf_mean,hlf_std



def main(argv):
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('sources', type=str, nargs='+')
    parser.add_argument('-o', '--output_dir', type=str, dest='output_dir', required=True)
    try:
        args = parser.parse_args(argv)
    except Exception:
        parser.print_usage()

    particle_mean, particle_std, hlf_mean, hlf_std = gen_std_stats(args.sources)
    
    o_path = os.path.abspath(args.output_dir)
    f = h5py.File(o_path,'w')
    f.create_dataset('particle_mean',data=particle_mean)
    f.create_dataset('particle_std', data=particle_std)
    f.create_dataset('hlf_mean', data=hlf_mean)
    f.create_dataset('hlf_std', data=hlf_std)
    f.close()

if __name__ == "__main__":
   main(sys.argv[1:])