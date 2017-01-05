import types
import numpy as np
from CMS_Deep_Learning.storage.archiving import DataProcedure

class DataIterator:
    def __init__(self, proc, num_samples=None, return_X=False, return_Y=True, accumilate=None, prediction_model=None):
        if (isinstance(proc, list)):
            first_data = proc[0].getData()
        else:
            first_data = proc.getData()
        if (isinstance(first_data, types.GeneratorType)):
            proc = first_data

        self.proc = proc
        self.num_samples = num_samples
        self.accumilate = accumilate
        self.prediction_model = prediction_model
        self.return_X = return_X
        self.return_Y = return_Y
        if (isinstance(proc, list)):
            if (False in [isinstance(p, DataProcedure) for p in proc]):
                raise ValueError("procedure list must contain only DataProcedures")
            # self.next = _listNext
            self.proc_itr = iter(self.proc)
            self.mode = "list"
        elif (isinstance(proc, types.GeneratorType)):
            if (num_samples == None):
                raise ValueError("num_samples must be passed along with procedure generator.")
            # self.next = _genNext
            self.mode = "generator"
        else:
            raise ValueError("Bad input.")
            # initialize(proc, num_samples=num_samples, accumilate=accumilate, prediction_model=prediction_model)

    def getLength(self):
        if (self.num_samples == None):
            num_samples = 0
            for p in self.proc:
                if (isinstance(p, DataProcedure)):
                    X, Y = p.getData()
                else:
                    X, Y = p
                if (not isinstance(Y, list)): Y = [Y]
                num_samples += Y[0].shape[0]
            self.num_samples = num_samples
        return self.num_samples

    def asList(self):
        X_out = None
        Y_out = None
        pred_out = None
        acc_out = None
        pos = 0
        for p in self.proc:
            X, Y = p.getData()

            if (not isinstance(Y, list)): Y = [Y]
            L = Y[0].shape[0]

            if (not isinstance(X, list)): X = [X]
            if (self.return_X):
                if (X_out == None): X_out = [[None] * self.getLength() for i in range(len(X))]
                # print([len(x) for x in X_out])
                # print([len(x) for x in X])
                for i, x in enumerate(X):
                    Xi_out = X_out[i]
                    # print(len(xi))
                    for j in range(L):
                        Xi_out[pos + j] = x[j]

            if (self.return_Y):
                if (Y_out == None): Y_out = [[None] * self.getLength() for i in range(len(Y))]
                for i, y in enumerate(Y):
                    Yi_out = Y_out[i]
                    for j in range(L):
                        Yi_out[pos + j] = y[j]

            if (self.prediction_model != None):
                if (pred_out == None): pred_out = [None] * self.getLength()
                pred = self.prediction_model.predict_on_batch(X)
                for j in range(L):
                    pred_out[pos + j] = pred[j]

            if (self.accumilate != None):
                if (acc_out == None): acc_out = [None] * self.getLength()
                acc = self.accumilate(X)
                for j in range(L):
                    acc_out[pos + j] = acc[j]

            pos += L
            print(pos, self.accumilate)  # ,acc_out))
        out = []
        if (X_out != None):
            for i, xo in enumerate(X_out):
                X_out[i] = np.array(xo)
            out.append(X_out)
        if (Y_out != None):
            for i, yo in enumerate(Y_out):
                Y_out[i] = np.array(yo)
            out.append(Y_out)
        if (pred_out != None):
            out.append(np.array(pred_out))
        if (acc_out != None):
            out.append(np.array(acc_out))
        return out

    # def _genNext():
    #    #N = num_samples/ba
    #    data = data.getData()
    #    for i in range(self.num_samples)
    #        yield next(data)
    #    return StopIteration()
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

        # def next(self):
        #    self.count += 1
        #    if(self.mode == 0):


class TrialIterator(DataIterator):
    def __init__(self, trial, data_type="val", return_X=False, return_Y=True, accumilate=None, return_prediction=False,
                 custom_objects={}):
        if (data_type == "val"):
            proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.val_procedure]
            num_samples = trial.nb_val_samples
        elif (data_type == "train"):
            proc = [DataProcedure.from_json(trial.archive_dir, t) for t in trial.train_procedure]
            num_samples = trial.samples_per_epoch
        else:
            raise ValueError("data_type must be either val or train but got %r" % data_type)
        model = None
        if (return_prediction):
            model = trial.compile(loadweights=True, custom_objects=custom_objects)
        DataIterator.__init__(self, proc, num_samples=num_samples, return_X=return_X, return_Y=return_Y,
                              accumilate=accumilate, prediction_model=model)
