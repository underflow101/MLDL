#####################################################################################
# quantOpt.py
#
# Dev. Dongwon Paek
# Description: Real example of converting keras weight file to .tflite file
#              Some functions are commented out for optional purpose
#####################################################################################

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model('./saved')
new_model.summary()

# tflite converter
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_types = [tf.uint8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
open("quant_model_3-1.tflite", "wb").write(tflite_model)

print("Model successfully converted into tflite file.")
