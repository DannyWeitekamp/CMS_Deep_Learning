import pandas as pd
import os,sys

def msgpack_assertMeta(filename, frames=None, redo=False):
    meta_out_file = filename.replace(".msg", ".meta")
    print(meta_out_file)
    meta_frames = None
    if(os.path.exists(meta_out_file) and not redo):
        meta_frames = pd.read_msgpack(meta_out_file)
    if(meta_frames == None):
        if(frames == None):
            print("Bulk reading .msg for metaData assertion. Be patient, reading in slices not supported.")
            print(filename)
            frames = pd.read_msgpack(filename)
        meta_frames = {"NumValues" : frames["NumValues"]}
           
    if(not os.path.exists(meta_out_file) or redo):
        pd.to_msgpack(meta_out_file, meta_frames)

    return meta_frames