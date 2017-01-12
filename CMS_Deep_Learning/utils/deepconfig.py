class deepconfig:
    def __init__(self, gpu='gpu', backend='theano'):
        import os
        import json
        t_config = {}
        open('/home/%s/.theanorc'%os.getenv('USER'),'w').write('[nvcc]\nfastmath=True\nflags =  -arch=sm_30\n[global]\n#mode=FAST_RUN\ndevice=%s\nfloatX=float32'%gpu)
        print "using",gpu
        k_config = json.loads( open('/home/%s/.keras/keras.json'%os.getenv('USER') ).read())
        k_config['backend'] = backend
        open('/home/%s/.keras/keras.json'%os.getenv('USER'),'w').write( json.dumps( k_config ))
        print "using",backend
