import glob
import json

import DatasetInserter
import Model
import numpy as np
import tensorflow as tf
files = glob.glob('data/*')

ds = DatasetInserter.dataSet("potData.json", "potData.csv", debug=True, save=True)
Classifier = DLM.classifierModel('ClassifierModel.keras', 1, 2)
Classifier.fitModelLoop(ds.data["X"], ds.getBinaryY(), 800)