import sys,os
from keras import backend as K
from keras.backend.common import _EPSILON
from keras.constraints import maxnorm
from keras.engine.topology import Layer
import keras
from keras.optimizers import Optimizer,clip_norm
import theano
import numpy as np
import types
from six import string_types

DEFAULT_MAPPINGS = {"Identity":lambda x:x}

if K.backend() == 'theano':
    def argsort_top_k(x, k):
        '''Returns the indicies of the k largest elements'''
        from theano import tensor as T
        x = K.flatten(x)
        drop = x.shape[0] - k
        # x = theano.printing.Print('drop', attrs=['shape'])(x)
        indicies = T.argsort(x)[drop:]
        return indicies

elif K.backend() == 'tensorflow':
    def argsort_top_k(x, k):
        '''Returns the indicies of the k largest elements'''
        import tensorflow as T
        return T.nn.top_k(x,k,sorted=True)[1]


def get_func(name, module):
    '''Get a function from its name and module path'''
    try:
        exec ("from " + module + " import " + name + " as prep_func")
    except ImportError:
        raise ValueError("Function %r does not exist in %r. \
            For best results functions should be importable and not locally defined." % (str(name), str(module)))
    return locals().get("prep_func", None)

class Sort(Layer):
   
    def __init__(self, mapping="Identity", nb_out=-1, initial_beta='random', **kwargs):
        if(isinstance(mapping,tuple) and len(mapping) == 2):
            mapping = get_func(*mapping)
        if(not isinstance(mapping,(string_types,types.FunctionType))):
            raise ValueError("Mapping must be str or function but got %r" % type(mapping))
        if(isinstance(mapping,string_types) and not mapping in DEFAULT_MAPPINGS):
            raise ValueError("Mapping %r not recognized" % mapping)
        
        self.mapping,self.mapping_name = (DEFAULT_MAPPINGS[mapping],mapping) if isinstance(mapping,string_types) else (mapping,(mapping.__name__,mapping.__module__))
        self.nb_out = nb_out
        self.initial_beta = initial_beta
        super(Sort, self).__init__(**kwargs)

    def build(self, input_shape):
        # print(input_shape)
        fake_shape = tuple([1 if x == None else x for x in input_shape])
        mapping_dim = K.eval(self.mapping(K.variable(np.zeros(fake_shape)))).shape[-1]
        self.mapping_dim = mapping_dim
        if(isinstance(self.initial_beta, string_types)):
            assert self.initial_beta in ["zeros", 'random'], "Initialization %r not recognized." % self.initial_beta
            initial_beta = np.zeros(mapping_dim) if self.initial_beta == 'zeros' else np.random.random(mapping_dim)
        else:
            initial_beta = self.initial_beta
        assert len(initial_beta) == mapping_dim, 'Beta initialization should have length %r but got length %r' % (mapping_dim,len(initial_beta))
        self.beta = K.variable(initial_beta)
        setattr(self.beta, "finite_only", True)
        print(self.__dict__)
        self.trainable_weights = [self.beta]
        
    def call(self, T, mask=None):
        import theano
        if(hasattr(self,'perturb_each_sample') and self.perturb_each_sample):
            # print(K.eval(self.beta))
            self.beta = theano.printing.Print("BETa", attrs=['shape'])(self.beta)
            # metrics = K.zeros_like(T,dtype=T.dtype )#K.batch_dot(self.mapping(T), self.beta)
            metrics = K.batch_dot(self.mapping(T), self.beta, axes=1)
        else:
            metrics = K.dot( self.mapping(T), K.reshape(self.beta, (self.beta.shape[-1],1)))

        #Resolve the number of sequence elements to keep
        k = K.variable(self.nb_out if self.nb_out != -1 else sys.maxint,dtype='int') 
        k = K.minimum(K.shape(T)[-2],k)
        # k = theano.printing.Print('k')(k)
        
        #Find from which indicies to grab from the input
        top_k_sort = lambda x: argsort_top_k(x,k)
        indicies = K.map_fn(top_k_sort,metrics)
        
        #Grab the sequence elements from the input
        get_subset = lambda x,y: K.gather(x,y)
        out = K.map_fn(get_subset,[T,indicies])
        
        # out = theano.printing.Print('out')(out)
        
        return out
        
    def get_output_shape_for(self, input_shape):
        return tuple([x if (i != 1 or self.nb_out == -1) else self.nb_out for i,x in enumerate(input_shape)])

    def get_config(self):
        base_config = Layer.get_config(self)
        config = {'mapping': self.mapping_name,
                  'initial_beta': self.initial_beta,
                  'nb_out': self.nb_out}
        return dict(list(base_config.items()) + list(config.items()))
    
class Perturbed_Sort(Sort):
    def __init__(self, parent, lr=.1, perturb_each_sample=False, **kwargs):
        self.parent = parent
        self.lr = K.variable(lr)
        self.perturb_each_sample = perturb_each_sample
        for key,val in parent.get_config().items():
            kwargs[key] = val
        kwargs['name'] = 'pert_' + kwargs['name']
        super(Perturbed_Sort, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = [self.parent.beta]

    def call(self, T, mask=None):
        if(self.perturb_each_sample):
            # self.beta = K.map_fn(lambda x:self.parent.beta + K.random_normal((self.parent.mapping_dim,),mean=0,std=self.lr,dtype=self.parent.beta.dtype) ,T)
            self.beta = self.parent.beta + K.random_normal((T.shape[0],self.parent.mapping_dim),mean=0,std=self.lr,dtype=self.parent.beta.dtype)
        else:
            dev = K.zeros((self.parent.mapping_dim,),dtype=self.parent.beta.dtype)
            loc = K.random_uniform((1,),low=0.0,high=self.parent.mapping_dim)
            dev = 
            # self.beta = self.parent.beta + K.random_normal((self.parent.mapping_dim,),mean=0,std=self.lr,dtype=self.parent.beta.dtype)
            self.beta = self.parent.beta + dev #+ K.random_normal((self.parent.mapping_dim,),mean=0,std=self.lr,dtype=self.parent.beta.dtype)
        return super(Perturbed_Sort, self).call(T,mask=mask)
    
class Finite_Differences(Optimizer):
    def __init__(self,model, main,pert, backup='adam', **kwargs):
        self.backup = keras.optimizers.get(backup)
        self.get_updates = self.backup.get_updates
        self.main = main
        self.pert= pert
        self.model = model
        #Monkey Patch Optimizer
        Optimizer.get_gradients = self.get_gradients
        super(Optimizer,self).__init__(**kwargs)
    
    def get_gradients(self, loss, params):
        '''A Monkey Patched version of get_gradients that handles finite differences'''
        finites = filter(lambda x: hasattr(x,"finite_only"), params)
        params = filter(lambda x: not hasattr(x,"finite_only"), params)
        finites = list(set(finites))
        # params = [x if hasattr(x,"finite_only") else None for x in params]
        # print(type(self.main.beta),type(self.pert.beta), (self.main.beta-self.pert.beta).shape)
        metric_dict = {key: val for key, val in zip(self.model.metrics_names[1:], self.model.metrics_tensors)}
        # print(metric_dict)

        self.pert.beta = theano.printing.Print("BETA")(self.pert.beta)
        weight_diffs = [self.pert.beta - self.main.beta]
        # weight_diffs = [x-p for x,p in zip(self.main.trainable_weights, self.pert.trainable_weights) if x in finites]
        loss_diff =  K.minimum(metric_dict[self.pert.name + "_loss"] - metric_dict[self.main.name + "_loss"],0.0)
        loss_diff = theano.printing.Print("loss_diff")(loss_diff)
        if (self.pert.perturb_each_sample):
            grads = [loss_diff / K.sum(x,axis=0) for x in weight_diffs]
        else:
            grads = [loss_diff/x for x in weight_diffs]
        # print(metric_dict)
        # print(weight_diffs)

        
        grads[0] = theano.printing.Print("grad")(grads[0])
        # weight_diffs[0] = theano.printing.Print("weights")(self.main.trainable_weights[0])
        # grads = [loss_diff]
        # grads = weight_diffs
        # print(self.model.metrics_names[1:])
        # print(self.model.metrics_tensors)
        # 
        print("FINITS", [(x.name,x.dtype) for x in finites])
        # print("Params", params)
        # print("Losses", [x.__dict__ for x in loss.owner.inputs[0].owner.inputs])
        # print(self.main)
        # print(self.pert)
        # loss = theano.printing.Print("LOSS")(loss)
        # self.model.metrics_tensors[0] = theano.printing.Print("1")(self.model.metrics_tensors[0])
        # self.model.metrics_tensors[1] = theano.printing.Print("2")(self.model.metrics_tensors[1])
        # K.print_tensor(loss,"LOSS")
        # theano.printing.debugprint(loss)
        grads = grads + K.gradients(loss, params)
        # print("GRADS LEN", len(grads))
        # grads = theano.printing.Print("grads")(grads[0])
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
            
    # def get_updates(self, params, constraints, loss):
    #     # grads = self.get_gradients(loss, params)
    #     print("MOOP")
    #     print(params)
    #     print(loss)
    #     print(constraints)
    #     return self.backup.get_updates(params, constraints, loss)#self.updates
    # 
        
