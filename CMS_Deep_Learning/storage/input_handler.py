from CMS_Deep_Learning.io import DataIterator, TrialIterator
from CMS_Deep_Learning.storage.archiving import KerasTrial
from six import string_types
import numpy as np

REQ_DICT = {"predictions": [['trial'], ['model', 'data'], ['model', 'X']],
            "characteristics": [['trial', 'accumilate'], ['model', 'data', 'accumilate'], ['model', 'X', 'accumilate']],
            "X": [['trial'], ['data']],
            "Y": [['trial'], ['data']],
            "model": [['trial']],
            "num_samples": [['trial']]}
ITERATOR_REQS = ['predictions', 'characteristics', 'X', 'Y', 'num_samples']


def assertModel(model, weights=None, loss='categorical_crossentropy', optimizer='rmsprop', custom_objects={}):
    '''Asserts that the inputs create a valid keras model and returns that model

        :param model: a keras Model or the path to a model .json
        :type model: str or Model
        :param weights: the model weights or path to the stored weights
        :type weights: str or weights
        :param loss: the loss function to compile the model with
        :type loss: str
        :param : the optimizer to compile the model with
        :type optimizer: str
        :param custom_objects: a dictionary of user defined classes
        :type custom_objects: dict of classes
        :returns: A compiled model
        '''
    from keras.engine.training import Model
    from keras.models import model_from_json
    import os, sys
    '''Takes a model and weights, path and weights, json_sting and weights, or compiled model
        and outputs a compiled model'''
    if (loss == None): loss = 'categorical_crossentropy'
    if (optimizer == None): optimizer = 'rmsprop'

    if (isinstance(model, string_types)):
        if (os.path.exists(model)):
            model_str = open(model, "r").read()
        else:
            model_str = model
        model = model_from_json(model_str, custom_objects=custom_objects)
    # If not compiled
    if not hasattr(model, 'test_function'):
        if (isinstance(weights, type(None))):
            raise ValueError("Cannot compile without weights")
        if (isinstance(weights, string_types) and os.path.exists(weights)):
            model.load_weights(weights)
        else:
            model.set_weights(weights)
    return model


def assertType(x, t):
    '''Asserts that x is of type t and raises an error if not'''
    assert isinstance(x, t), "expected %r but got type %r" % (t, type(x))


def _checkAndAssert(data_dict, data_to_check):
    '''A helper function for simple_grab that checks and asserts the correct data types'''
    if ("model" in data_to_check):
        data_dict['model'] = assertModel(data_dict['model'],
                                         weights=data_dict.get('weights', None),
                                         loss=data_dict.get('loss', None),
                                         optimizer=data_dict.get('optimizer', None),
                                         custom_objects=data_dict.get('custom_objects', {})
                                         )
    if ("trial" in data_to_check): assertType(data_dict['trial'], KerasTrial)
    if ("X" in data_to_check): assertType(data_dict['X'], (np.ndarray, list, tuple))
    if ("Y" in data_to_check): assertType(data_dict['Y'], (np.ndarray, list, tuple))
    if ("predictions" in data_to_check): assertType(data_dict['predictions'], np.ndarray)
    if ("num_samples" in data_to_check): assertType(data_dict['num_samples'], int)

    return data_dict


def _call_iters(data_dict, to_return, sat_dict):
    '''A helper function for simple_grab that calls the DataIterators if necessary'''
    if (len(set.intersection(set(to_return), set(ITERATOR_REQS))) != 0):
        to_get = [req for req in to_return if req in ITERATOR_REQS and not req == sat_dict[req]]
        if (len(to_get) > 0):
            data_keys = []
            if ('X' in to_get): data_keys.append(data_dict.get('input_keys', 'X'))
            if ('Y' in to_get): data_keys.append(data_dict.get('label_keys', 'Y'))
            accumilate = data_dict.get('accumilate', None)  # if('accumilate' in to_get) else None
            if (sat_dict[to_get[0]][0] == 'trial'):
                dItr = TrialIterator(data_dict['trial'],
                                     data_keys=data_keys,
                                     input_keys=data_dict.get('input_keys'),
                                     label_keys=data_dict.get('label_keys'),
                                     return_prediction='predictions' in to_get,
                                     accumilate=accumilate)
                out = dItr.as_list(verbose=0)
            else:
                dItr = DataIterator(data_dict.get('data', None),
                                    data_keys=data_keys,
                                    num_samples=data_dict.get('num_samples', None),
                                    input_keys=data_dict.get('input_keys', 'X'),
                                    label_keys=data_dict.get('label_keys', 'Y'),
                                    prediction_model=data_dict.get('model', None),
                                    accumilate=accumilate)
                out = dItr.as_list(verbose=0)
            for i, key in enumerate(to_get):
                data_dict[key] = out[i]
    return data_dict


def simple_grab(to_return, data_dict={}, **kargs):
    '''Returns the data requested in to_return given that the data can be found/derived from the given inputs.
        for example one can derive predictions from a model path, weights path, and X (input data).
         Input information includes ['trial', 'model,'data,'X','Y', accumilate,'predictions', 'characteristics', 'X', 'Y', 'model', 'num_samples'].
         outputs include ['predictions','characteristics', 'X', 'Y', 'model', 'num_samples'].



        :param to_return: A set of requirements, options: predictions,X,Y,model,num_samples
        :returns: the data requested in to_return'''

    if (len(kargs) != 0): data_dict = kargs
    data_to_check = set([])
    sat_dict = {}
    for req in to_return:
        if not req in REQ_DICT:
            raise ValueError("Requirement %r not recognized" % req)
        satisfiers = REQ_DICT[req]
        ok = [not False in [x in data_dict for x in sat] \
              for sat in satisfiers]
        if (not req in data_dict and not True in ok):
            raise ValueError('To handle requirement %r need (%s) or %s' % \
                             (req, req, ' or '.join(['(' + ",".join(x) + ')' for x in satisfiers])))
        satisfier = req if req in data_dict else satisfiers[ok.index(True)]
        sat_dict[req] = satisfier
        for x in satisfier:
            data_to_check.add(x)

    data_dict = _checkAndAssert(data_dict, data_to_check)
    data_dict = _call_iters(data_dict, to_return, sat_dict)
    # out = []

    return tuple([data_dict[r] for r in to_return])