import pandas as pd
import os,sys

def msgpack_assertMeta(filename, frames=None, redo=False):
    meta_out_file = filename.replace(".msg", ".meta")
    print(meta_out_file)
    meta_frames = None
    if(os.path.exists(meta_out_file) and not redo):
        #Need to check for latin encodings due to weird pandas default
        try:
            meta_frames = pd.read_msgpack(meta_out_file)
        except UnicodeDecodeError as e:
            meta_frames = pd.read_msgpack(meta_out_file, encoding='latin-1')
    if(meta_frames == None):
        if(frames == None):
            print("Bulk reading .msg for metaData assertion. Be patient, reading in slices not supported.")
            print(filename)
            try:
                frames = pd.read_msgpack(filename)
            except UnicodeDecodeError as e:
                frames = pd.read_msgpack(filename, encoding='latin-1')
        meta_frames = {"NumValues" : frames["NumValues"]}
           
    if(not os.path.exists(meta_out_file) or redo):
        pd.to_msgpack(meta_out_file, meta_frames)

    return meta_frames