import warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.use('Agg')

from nilmtk import DataSet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from rnndisaggregator import RNNDisaggregator
from nilmtk.datastore import HDFDataStore
import sys

appl = ['fridge', 'washing_machine', 'television']

#from nilmtk.dataset_converters import convert_ukdale
#convert_ukdale('./data/UKDALE', './data/UKDALE/ukdale.h5')  # Skip if we already have the data in .h5 file
if len(sys.argv) < 2:
    print("Error in arguments usage\n")
    exit()

APPLIANCE = sys.argv[1]

if APPLIANCE not in appl:
    print("Error in type of applicance\n")
    exit()

DATASET = '../data/ENEA/enea.h5'
MODEL = '../data/ENEA/model-lstm-' + APPLIANCE + 'enea.h5'
DISAG = '../data/ENEA/disag-lstm-' + APPLIANCE + 'out.h5'
TRAIN_BUILDING = 1
TEST_BUILDING = 1

train = DataSet(DATASET)
train.set_window(start="2017-03-11", end="2017-08-31") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters

rnn = RNNDisaggregator()

train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()[APPLIANCE] # The kettle meter that is used as a training target

#print("------ TRAINING ------")
#rnn.train(train_mains, train_meter, epochs=5, sample_period=1)
#rnn.export_model(MODEL)

rnn.import_model(MODEL)
test = DataSet(DATASET)
test.set_window(start="2017-09-01", end="2017-10-31")
test_elec = test.buildings[TEST_BUILDING].elec
test_mains = test_elec.mains()

output = HDFDataStore(DISAG, 'w')

print("------ TESTING ------")
rnn.disaggregate(test_mains, output, train_meter, sample_period=1)

result = DataSet(DISAG)
res_elec = result.buildings[TEST_BUILDING].elec
predicted = res_elec[APPLIANCE]
ground_truth = test_elec[APPLIANCE]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(predicted.power_series_all_data(), label='predicted')
ax.plot(ground_truth.power_series_all_data(), label='ground truth')
plt.xlim('2017-09-18 00:00:00', '2017-09-18 23:59:59')
plt.xlabel('Time')
plt.ylabel('Power [W]')
plt.title(APPLIANCE + ' Disaggregation')
myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.legend()
plt.savefig(APPLIANCE + "_lstm.png")

import metrics
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
