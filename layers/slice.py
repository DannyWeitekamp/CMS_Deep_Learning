from keras.engine.topology import Layer

class Slice(Layer):
    '''Applies a slice to input data and outputs the results, does not slice along the batch axis
        #Arguments:
            split_str -- a string like "[A:B:C, D:E:F, ... ,]" designating the splits to make on the input data
                where each argument has one of the usual forms start:stop:stride, start:stop, :stop, start:, etc.
            Note: Splitting on the batch axis is protected against so [:,4:5] is really [:,:,4:5]
    '''
    def __init__(self, split_str, **kwargs):
        super(Slice, self).__init__(**kwargs)

        def _decodeSlice(x):
            # print(x)
            if(len(x) < 1 or len(x) > 3):
                raise ValueError("Not possible slice")
            if(len(x) == 1):
                if(x[0] == ''): raise ValueError("Not possible slice")
                return (int(x[0]),int(x[0]),1)
            start = 0 if x[0] == '' else int(x[0])
            end = None if x[1] == '' else int(x[1])
            step = 1 if(len(x) == 2 or x[2] == '') else int(x[2])
            # print(start,end,start)
            return (start, end, step)
            
        self.split_str = split_str
        self.process_split_str = split_str.replace('[', '[:,', 1)
        terms = split_str.replace('[', '').replace(']', '').split(',')
        self.splits = [ _decodeSlice(t.split(':')) for i, t in enumerate(terms)]
        # print(args)

    def call(self, T, mask=None):
        # start =  K.variable(self.start, dtype=np.int32)
        # stop =  K.variable(self.stop, dtype=np.int32)
        # T_slice = T[:,:,self.start:self.stop]
        exec('T_slice = T' + self.process_split_str)
        return T_slice
    def get_output_shape_for(self, input_shape):
        l = list(input_shape)
        for i in range(1,len(input_shape)):
            start,end,step = self.splits[i-1]
            if(end == None): end = input_shape[i]
            l[i] = max((end-start),1) // step
        # return (input_shape[0], self.output_dim)
        # return (input_shape[0], input_shape[1], self.stop-self.start)
        return tuple(l)
    def get_config(self):
        base_config = Layer.get_config(self)
        config = {'split_str': self.split_str}
        return dict(list(base_config.items()) + list(config.items()))


# s = Slice("[:2 ,3:4 ,1 , 25:, 0:10:2]")
# out = s.get_output_shape_for((100,5,3, 50, 50,50))
# print(out)