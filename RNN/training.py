import warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl
from nilmtk import DataSet
import matplotlib.pyplot as plt
from rnndisaggregator import RNNDisaggregator
from nilmtk.datastore import HDFDataStore

mpl.use('Agg')

#from nilmtk.dataset_converters import convert_ukdale
#convert_ukdale('./data/UKDALE', './data/UKDALE/ukdale.h5')  # Skip if we already have the data in .h5 file

TRAIN_BUILDING = 1
TEST_BUILDING = 5
"""
train = DataSet('../data/UKDALE/ukdale.h5')
train.set_window(start="2013-04-12", end="2015-07-01") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters

rnn = RNNDisaggregator()

train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()['fridge'] # The kettle meter that is used as a training target

print("------ TRAINING ------")
rnn.train(train_mains, train_meter, epochs=5, sample_period=6)
rnn.export_model("../data/UKDALE/model-ukdale.h5")
"""
rnn = RNNDisaggregator()

rnn.import_model("../data/UKDALE/model-ukdale.h5")
test = DataSet('../data/UKDALE/ukdale.h5')
test.set_window(start="2014-06-29", end="2014-10-21")
test_elec = test.buildings[TEST_BUILDING].elec
test_mains = test_elec.mains()

disag_filename = '../data/UKDALE/disag-out.h5' # The filename of the resulting datastore
from nilmtk.datastore import HDFDataStore
output = HDFDataStore(disag_filename, 'w')

print("------ TESTING ------")
rnn.disaggregate(test_mains, output, train_meter, sample_period=6)

result = DataSet(disag_filename)
res_elec = result.buildings[TEST_BUILDING].elec
predicted = res_elec['fridge']
ground_truth = test_elec['fridge']

predicted.power_series_all_data().plot()
ground_truth.power_series_all_data().plot()
plt.xlim('2014-08-22 00:00:00', '2014-08-22 23:59:00')
plt.savefig("fridge.png")

import metrics
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
