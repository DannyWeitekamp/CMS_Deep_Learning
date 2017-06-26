from CMS_Deep_Learning.storage.iterators import DataIterator,TrialIterator

REQ_DICT = {"predictions": [['trial'], ['model', 'data'], ['model', 'X']],
            "characteristics": [['trial', 'accumilate'], ['model', 'data', 'accumilate'], ['model', 'X', 'accumilate']],
            "X": [['trial'], ['data']],
            "Y": [['trial'], ['data']],
            "model": [['trial']],
            "num_samples": [['trial']]}
ITERATOR_REQS = ['predictions', 'characteristics', 'X', 'Y', 'num_samples']


def assertModel(model, weights=None, loss='categorical_crossentropy', optimizer='rmsprop', custom_objects={}):
    from keras.engine.training import Model
    from keras.models import model_from_json
    import os, sys
    '''Takes a model and weights, path and weights, json_sting and weights, or compiled model
        and outputs a compiled model'''
    if (loss == None): loss = 'categorical_crossentropy'
    if (optimizer == None): optimizer = 'rmsprop'

    if (isinstance(model, str)):
        if (os.path.exists(model)):
            model_str = open(model, "r").read()
        else:
            model_str = model
        model = model_from_json(model_str, custom_objects=custom_objects)
    # If not compiled
    if not hasattr(model, 'test_function'):
        if (isinstance(weights, type(None))):
            raise ValueError("Cannot compile without weights")
        if (isinstance(weights, str) and os.path.exists(weights)):
            model.load_weights(weights)
        else:
            model.set_weights(weights)
    return model


def assertType(x, t):
    '''Asserts that x is of type t and raises an error if not'''
    assert isinstance(x, t), "expected %r but got type %r" % (t, type(x))


def inputHandler(req_info):
    '''Returns an inputHandler function with a set of requirements. The inputHandler function will try
       to derive the required information from the given information, for example it can derive predictions
       from a model path,weights path, and X (input data)

       :param req_info: A set of requirements, options: predictions,X,Y,model,num_samples

       :returns: an inputHandler function with input options predictions,X,Y,model,num_samples,weights,trial,data'''

    def f(data_dict):
        data_to_check = set([])
        sat_dict = {}
        for req in req_info:
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
        if ("model" in data_to_check):
            data_dict['model'] = assertModel(data_dict['model'],
                                             weights=data_dict.get('weights', None),
                                             loss=data_dict.get('loss', None),
                                             optimizer=data_dict.get('optimizer', None),
                                             custom_objects=data_dict.get('custom_objects', {})
                                             )
        if ("trial" in data_to_check): assertType(data_dict['trial'], KerasTrial)
        if ("X" in data_to_check): assertType(data_dict['X'], np.ndarray)
        if ("Y" in data_to_check): assertType(data_dict['Y'], np.ndarray)
        if ("predictions" in data_to_check): assertType(data_dict['predictions'], np.ndarray)
        if ("num_samples" in data_to_check): assertType(data_dict['num_samples'], int)

        out = []
        if (len(set.intersection(set(req_info), set(ITERATOR_REQS))) != 0):
            to_get = [req for req in req_info if not req == sat_dict[req]]
            data_keys = []
            if ('X' in to_get): data_keys.append("X")
            if ('Y' in to_get): data_keys.append("Y")
            # TODO: if('accumilation' in to_get): accumilate = data_dict['']
            print("TO_GET", to_get)
            accumilate = data_dict.get('accumilate', None)  # if('accumilate' in to_get) else None
            return_prediction = True if ('predictions' in to_get) else False
            if (sat_dict[to_get[0]][0] == 'trial'):
                print("ACCUM_TR", accumilate)
                dItr = TrialIterator(data_dict['trial'],
                                     data_keys=data_keys,
                                     return_prediction=return_prediction,
                                     accumilate=accumilate)
                out = dItr.asList(verbose=0)
            else:
                print("ACCUM_NTR", accumilate)
                dItr = DataIterator(data_dict.get('data', None),
                                    data_keys=data_keys,
                                    num_samples=data_dict.get('num_samples', None),
                                    prediction_model=data_dict.get('model', None),
                                    accumilate=accumilate)
                out = dItr.asList(verbose=0)
        return tuple(list(out) + [data_dict[r] for r in req_info if not r in ITERATOR_REQS])

    return f


g = inputHandler(['Y', "predictions"])
# g({"data":0, "labels":0})