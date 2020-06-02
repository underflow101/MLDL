#####################################################################################
# quantOpt.py
#
# Dev. Dongwon Paek
# Description: Real example of converting keras weight file to .tflite file
#####################################################################################

import tensorflow as tf
import numpy as np

saved_model_dir = "./model.pb"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)
open("quant_model.tflite", "wb").write(tflite_quant_model)

print("Model successfully converted into tflite file.")