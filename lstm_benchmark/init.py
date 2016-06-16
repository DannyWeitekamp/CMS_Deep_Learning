#Init
import sys, os
if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.callbacks import SmartCheckpoint
name = "lstm_benchmark"
max_features = 20000
max_length = 5
embedding_dim = 256
checkpoint = SmartCheckpoint(name,
    monitor='val_acc',
    verbose=1,
    save_best_only=True)
