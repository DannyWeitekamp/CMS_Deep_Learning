import types
import sys
import numpy as np
from CMS_Deep_Learning.storage.archiving import DataProcedure
if(sys.version_info[0] > 2):
    from inspect import signature
    getNumParams = lambda f: len(signature(f).parameters)
else:
    from inspect import getargspec
    getNumParams = lambda f: len(getargspec(f)[0])


class DataIterator:
    def __init__(self, proc, num_samples=None, data_keys=[], input_key="X", label_key="Y",accumilate=None, prediction_model=None):
        if (isinstance(proc, list)):
            first_data = proc[0].get_data()
        else:
            first_data = proc.get_data()
        if (isinstance(first_data, types.GeneratorType)):
            proc = first_data

        self.proc = proc
        self.num_samples = num_samples
        self.accumilate = accumilate
        self.prediction_model = prediction_model
        self.data_keys= data_keys
        self.input_key= input_key
        self.label_key= label_key
        if (isinstance(proc, list)):
            if (False in [isinstance(p, DataProcedure) for p in proc]):
                raise ValueError("procedure list must contain only DataProcedures")
            self.proc_itr = iter(self.proc)
            self.mode = "list"
        elif (isinstance(proc, types.GeneratorType)):
            if (num_samples == None):
                raise ValueError("num_samples must be passed along with procedure generator.")
            self.mode = "generator"
        else:
            raise ValueError("Bad input.")

    def getLength(self,verbose=0):
        if (self.num_samples == None):
            num_samples = 0
            for p in self.proc:
                if (isinstance(p, DataProcedure)):
                    out = p.get_data(verbose=verbose, data_keys=[self.label_key])
                else:
                    out = p[0]
                Z = out[0]
                if (not isinstance(Z, list)): Z = [Z]
                num_samples += Z[0].shape[0]
            self.num_samples = num_samples
        return self.num_samples

    def asList(self,verbose=0):
        if (self.accumilate != None): num_params = getNumParams(self.accumilate)
        x_required = self.prediction_model != None and self.accumilate != None
        y_required = self.accumilate != None and num_params > 1
        
        union_keys = self.data_keys
        if(x_required):
            if(not self.input_key in union_keys):
                union_keys.append(self.input_key)
            input_index = union_keys.index(self.input_key)
        if(y_required):
            if(not self.label_key in union_keys):
                union_keys.append(self.label_key)
            label_index = union_keys.index(self.label_key)

        samples_outs = [None] * len(union_keys)
        
        pred_out = None
        acc_out = None
        pos = 0
        for p in self.proc:
            out = p.get_data(data_keys=union_keys,verbose=verbose)
            
            for i,Z in enumerate(out):
                if (not isinstance(Z, list)): Z = [Z]
                L = Z[0].shape[0]

                if (samples_outs[i] == None): samples_outs[i]= [[None] * self.getLength(verbose=verbose) for i in range(len(Z))]
                for j, z in enumerate(Z):
                    Zj_out = samples_outs[i][j]
                    for k in range(L):
                        Zj_out[pos + k] = z[k]

            if(x_required):
                X = samples_outs[input_index]
            if(y_required):
                Y = samples_outs[label_index]
            if (self.prediction_model != None):
                if (pred_out == None): pred_out = [None] * self.getLength(verbose=verbose)
                pred = self.prediction_model.predict_on_batch(X)
                for j in range(L):
                    pred_out[pos + j] = pred[j]

            if (self.accumilate != None):
                if (acc_out == None): acc_out = [None] * self.getLength(verbose=verbose)
                if (num_params == 1):
                    acc = self.accumilate(X)
                else:
                    acc = self.accumilate(X, Y)
                for j in range(L):
                    acc_out[pos + j] = acc[j]

            pos += L
        out = []
        for key in self.data_keys:
            Z_out = samples_outs[union_keys.index(key)]
            if (Z_out != None):
                for j, zo in enumerate(Z_out):
                    Z_out[j] = np.array(zo)
                out.append(Z_out)
        if (pred_out != None):
            out.append(np.array(pred_out))
        if (acc_out != None):
            out.append(np.array(acc_out))
        return out

    '''
    def _listNext():
        for p in self.proc:
            X,Y = p.getData()
            pred = self.prediction_model.predict_on_batch(X) if self.prediction_model != None else None
            acc = self.accumilate(X) if self.accumilate != None else None
            for  in
                yield next(self.proc)
        return StopIteration()
    '''

    def __iter__(self):
        return self



class TrialIterator(DataIterator):
    def __init__(self, trial, data_type="val", data_keys=[], accumilate=None, return_prediction=False,
                 custom_objects={}):
        if (data_type == "val"):
            proc = trial.get_val()
            num_samples = trial.nb_val_samples
        elif (data_type == "train"):
            proc = trial.get_train()
            num_samples = trial.samples_per_epoch
        else:
            raise ValueError("data_type must be either val or train but got %r" % data_type)
        model = None
        if (return_prediction):
            model = trial.compile(loadweights=True, custom_objects=custom_objects)
        DataIterator.__init__(self, proc, num_samples=num_samples, data_keys=data_keys,
                              accumilate=accumilate, prediction_model=model)
