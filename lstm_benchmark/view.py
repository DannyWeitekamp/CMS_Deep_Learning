#View
from CMS_SURF_2016.utils.metrics import plot_history
import json
from keras.callbacks import History
if 'checkpoint' in locals():
	hist = checkpoint.histobj	
	plot_history([("lstm_benchmark",hist)]);
else:
	print("Must initialize first")