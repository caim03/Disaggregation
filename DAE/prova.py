
import warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.use('Agg')
from nilmtk import DataSet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from daedisaggregator import DAEDisaggregator
from nilmtk.datastore import HDFDataStore
#from nilmtk.dataset_converters import convert_ukdale
#convert_ukdale('./data/UKDALE', './data/UKDALE/ukdale.h5')  # Skip if we already have the data in .h5 file

SEQUENCE = 256
TRAIN_BUILDING = 1
TEST_BUILDING = 2

train = DataSet('../data/UKDALE/ukdale.h5')
train.set_window(start="2013-04-12", end="2015-07-01") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters
dae = DAEDisaggregator(SEQUENCE)
train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()['fridge'] # The kettle meter that is used as a training target

print("------ TRAINING ------")
#dae.train(train_mains, train_meter, epochs=50, sample_period=6)
#dae.export_model("../data/UKDALE/dae-ukdale.h5")
dae.import_model("../data/UKDALE/dae-ukdale.h5")

test = DataSet('../data/UKDALE/ukdale.h5')
test.set_window(start="2013-05-22", end="2013-09-24")
test_elec = test.buildings[TEST_BUILDING].elec
test_mains = test_elec.mains()
disag_filename = '../data/UKDALE/disag-dae-out.h5' # The filename of the resulting datastore
from nilmtk.datastore import HDFDataStore
output = HDFDataStore(disag_filename, 'w')

print("------ TESTING ------")
dae.disaggregate(test_mains, output, train_meter, sample_period=6)
result = DataSet(disag_filename)
res_elec = result.buildings[TEST_BUILDING].elec
predicted = res_elec['fridge']
ground_truth = test_elec['fridge']

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(predicted.power_series_all_data(), label='predicted')
ax.plot(ground_truth.power_series_all_data(), label='ground truth')
plt.xlim('2013-06-22 00:00:00', '2013-06-22 23:59:00')
plt.xlabel('Time [Hours]')
plt.ylabel('Power [W]')
plt.title('Fridge Disaggregation')
myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.legend()
plt.savefig("fridge_dae.png")

import metrics
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
