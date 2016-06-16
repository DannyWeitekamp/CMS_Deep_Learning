if __package__ is None:
	import sys, os
	sys.path.append(os.path.realpath("../../"))
from CMS_SURF_2016.utils.metrics import plot_history
import json
from keras.callbacks import History


hist = None
if 'checkpoint' in locals():
	if hasattr(checkpoint, "histobj"):
		hist = checkpoint.histobj	
	else:
		try:
			histDict = json.load(open( self.historyFilename, "rb" ))
			hist = History()
			hist.history = histDict
			print('Sucessfully loaded history at ' + self.historyFilename)
		except (IOError, EOFError):
			print('Failed to load history at ' + self.historyFilename)
	show_history([("lstm_benchmark",hist)]);