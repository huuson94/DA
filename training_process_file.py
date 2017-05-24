import pprint, pickle
import os

def read_origin_params(model):
    pkl_file = open(model+ ".pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data['param values']

def read_params(training_params_path):
    pkl_file = open(training_params_path + ".pkl", 'rb')
    training_params = pickle.load(pkl_file)
    pkl_file.close()
    return training_params

def save_params(params, training_params_path):
    filename = training_params_path + ".pkl"

    if not os.path.exists(os.path.dirname(filename)):
    	try:
            os.makedirs(os.path.dirname(filename))
	except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
            	raise

    training_params = open(filename, 'wb')
    pickle.dump(params, training_params)
    training_params.close()

def save_history(training_history, training_history_path):
    training_history_file = open(training_history_path+".pkl", 'wb')
    pickle.dump(training_history, training_history_file)
    training_history_file.close


