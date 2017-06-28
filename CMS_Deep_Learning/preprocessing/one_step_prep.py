import argparse,sys,os,delphes_parser
from CMS_Deep_Learning.preprocessing.preprocessing import preprocessFromPandas_label_dir_pairs, ObjectProfile,strideFromTargetSize,start_num_fromSplits,procsFrom_label_dir_pairs
from CMS_Deep_Learning.storage.batch import batchAssertArchived

DEFAULT_OBSERV_TYPES = ['E/c', 'Px', 'Py', 'Pz', 'PT_ET', 'Eta', 'Phi',
                        "MaxLepDeltaEta", "MaxLepDeltaPhi", 'MaxLepDeltaR', 'MaxLepKt', 'MaxLepAntiKt',
                        "METDeltaEta", "METDeltaPhi", 'METDeltaR', 'METKt', 'METAntiKt',
                        'Charge', 'X', 'Y', 'Z',
                        'Dxy', 'Ehad', 'Eem', 'MuIso', 'EleIso', 'ChHadIso', 'NeuHadIso', 'GammaIso', "ObjFt1",
                        "ObjFt2", "ObjFt3"]
DEFAULT_PHOTON_MAX = 100
DEFAULT_NEUTRAL_MAX = 100
DEFAULT_CHARGED_MAX = 100


def generatePandas(sources, num_samples,num_processes):
    for source in sources:
        delphes_parser.main(source, ['-n', num_samples, '-p', num_processes])

def applyPreprocessing(sources,
                       num_samples,
                       out_dir,
                       num_processes,
                       clean_pandas=False,
                       clean_archive=False,
                       size=250,
                       single_list=True,
                       sort_columns=["MaxLepDeltaR"],
                       sort_ascending=False,
                       photon_max=DEFAULT_PHOTON_MAX,
                       neutral_max=DEFAULT_NEUTRAL_MAX,
                       charged_max=DEFAULT_CHARGED_MAX,
                       ):
    #Run Final processing
    label_dir_pairs = [(s.split("/")[-1], s) for s in sources]
    print(label_dir_pairs)
    
    object_profiles = [
        # ObjectProfile("Photon", -1, pre_sort_columns=["PT_ET"], pre_sort_ascending=False, sort_columns=[sort_on], sort_ascending=False, addColumns={"ObjType":3}),
        ObjectProfile("EFlowPhoton", photon_max, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                      sort_columns=sort_columns, sort_ascending=sort_ascending,
                      addColumns={"ObjFt1": -1, "ObjFt2": -1, "ObjFt3": -1}),
        ObjectProfile("EFlowNeutralHadron", neutral_max, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                      sort_columns=sort_columns, sort_ascending=sort_ascending,
                      addColumns={"ObjFt1": -1, "ObjFt2": -1, "ObjFt3": 1}),
        ObjectProfile("EFlowTrack", charged_max, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                      sort_columns=sort_columns, sort_ascending=sort_ascending,
                      addColumns={"ObjFt1": -1, "ObjFt2": 1, "ObjFt3": -1}),
        ObjectProfile("Electron", 8, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                      sort_columns=sort_columns,
                      sort_ascending=sort_ascending, addColumns={"ObjFt1": -1, "ObjFt2": 1, "ObjFt3": 1}),
        ObjectProfile("MuonTight", 8, pre_sort_columns=["PT_ET"], pre_sort_ascending=False,
                      sort_columns=sort_columns,
                      sort_ascending=sort_ascending, addColumns={"ObjFt1": 1, "ObjFt2": -1, "ObjFt3": -1}),
        ObjectProfile("MissingET", 1, addColumns={"ObjFt1": 1, "ObjFt2": -1, "ObjFt3": 1}), ]
    
    temp_archive = "/".join([out_dir,'temp_archive'])
    if not os.path.exists(temp_archive):
        os.mkdir(temp_archive)
    
    stride = strideFromTargetSize(object_profiles, label_dir_pairs, DEFAULT_OBSERV_TYPES, megabytes=size)
    print(stride)
    
    #Here we are essentially creating 
    dps = procsFrom_label_dir_pairs(0,
                                    num_samples,
                                    stride,
                                    temp_archive,
                                    label_dir_pairs,
                                    object_profiles,
                                    DEFAULT_OBSERV_TYPES,
                                    single_list=single_list,
                                    sort_columns=sort_columns,
                                    sort_ascending=sort_ascending,
                                    verbose=0)
    batchAssertArchived(dps,num_processes=num_processes)
    


def _checkDir(dir):
    out_dir = os.path.abspath(dir)
    if (not os.path.exists(out_dir)):
        raise IOError("no such directory %r" % out_dir)
    return out_dir
# def _checkInteger(inp):
#     if (inp == ''): inp= None
#     return int(inp)

def main(argv):
    parser = argparse.ArgumentParser(description='Convert ROOT data to numpy arrays stored as HDF5 for Machine Learning use.')
    parser.add_argument('sources', type=str, nargs='+')
    parser.add_argument('-o', '--output_dir', type=str, dest='output_dir', required=True)
    parser.add_argument('-n','--num_samples', metavar='N', type=int, dest='num_samples',required=True)
    parser.add_argument('-p','--num_processes', metavar='N', type=int, dest='num_processes', default=1)
    parser.add_argument('-s','--size', metavar='N', type=int, default=250)
    parser.add_argument('--clean_pandas',action='store_true',default=False)
    parser.add_argument('--clean_archive',action='store_true',default=False)
    parser.add_argument('--skip_parse',action='store_true',default=False)
    
    parser.add_argument('--photon_max',metavar='N', type=int, default=DEFAULT_PHOTON_MAX)
    parser.add_argument('--neutral_max',metavar='N', type=int, default=DEFAULT_NEUTRAL_MAX)
    parser.add_argument('--charged_max',metavar='N', type=int, default=DEFAULT_CHARGED_MAX)
    # parser.print_usage()
    try:
        args = parser.parse_args(argv)
    except Exception:
        parser.print_usage()
        
    sources = [_checkDir(s) for s in args.sources]
    print(sources)
    print(args.clean_archive)
    print(args.clean_pandas)
    print(args.output_dir)
    print(args.num_samples)
    

    if(not args.skip_parse):
        generatePandas(sources,
             args.num_samples,
             args.num_processes)

    applyPreprocessing(sources,
         args.num_samples,
         args.output_dir,
         args.num_processes,
         clean_pandas=args.clean_pandas,
         clean_archive=args.clean_archive,
         size=args.size)
    
    # print(num_samples)
    

if __name__ == "__main__":
    main(sys.argv[1:])