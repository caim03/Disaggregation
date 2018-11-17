import warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000

from nilmtk import DataSet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from daedisaggregator import DAEDisaggregator
#from daedisaggregator2 import DAEDisaggregator
from nilmtk.datastore import HDFDataStore
import pandas as pd
import sys

appl = ['fridge', 'washing machine', 'television']

if len(sys.argv) < 4:
    print("Error in arguments usage\n")
    exit()

APPLIANCE = sys.argv[1]
if APPLIANCE not in appl:
    print("Error in type of applicance\n")
    exit()

if sys.argv[2] == 'True':
    TRAINING = True
else:
    TRAINING = False

if sys.argv[3] == 'fine_tune':
    FINE_TUNING = True
else:
    FINE_TUNING = False

DATASET = '../data/UKDALE/ukdale.h5'
MODEL = '../data/UKDALE/model-dae-' + APPLIANCE + 'ukdale.h5'
DISAG = '../data/UKDALE/disag-dae-' + APPLIANCE + 'out.h5'
UKDALE_MODEL = '../data/UKDALE/model-dae-washing machine-ukdale.h5'
TRAIN_BUILDING = 1
TEST_BUILDING = 2
SEQUENCE = 256

START_TEST = "2013-05-22"
END_TEST = "2013-09-24"

train = DataSet(DATASET)
train.set_window(start="2013-04-12", end="2015-07-01") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters

dae = DAEDisaggregator(SEQUENCE, FINE_TUNING)

if FINE_TUNING:
    print("------ FINE TUNING ------")
    dae.fine_tuning(UKDALE_MODEL)

train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()[APPLIANCE] # The kettle meter that is used as a training target

if TRAINING:
    print("------ TRAINING ------")
    dae.train(train_mains, train_meter, epochs=10, sample_period=1)
    dae.export_model(MODEL)
else:
    print("------ IMPORT MODEL ------")
    dae.import_model(UKDALE_MODEL)

# dae.import_model("../data/UKDALE/dae-ukdale.h5")
test = DataSet(DATASET)
test.set_window(start=START_TEST, end=END_TEST)
test_elec = test.buildings[TEST_BUILDING].elec
test_mains = test_elec.mains()

from nilmtk.datastore import HDFDataStore
output = HDFDataStore(DISAG, 'w')

print("------ TESTING ------")
dae.disaggregate(test_mains, output, train_meter, sample_period=1)

result = DataSet(DISAG)
res_elec = result.buildings[TEST_BUILDING].elec
predicted = res_elec[APPLIANCE]
ground_truth = test_elec[APPLIANCE]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(ground_truth.power_series_all_data(), label='ground truth')
ax.plot(predicted.power_series_all_data(), label='predicted')
plt.xlim('2013-06-02 00:00:00', '2013-06-02 23:59:59')
#plt.ylim(0, 300)
plt.xlabel('Time')
plt.ylabel('Power [W]')
plt.title(APPLIANCE + ' Disaggregation')
myFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.legend()
plt.savefig(APPLIANCE + "_dae.png")


import metrics
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
print("============ List of percentages for every days\n")
date_series = pd.date_range(start=START_TEST, end=END_TEST, freq='D')

metrics.daily_relative_consume(predicted, ground_truth, test_mains, date_series).to_csv('Percentages_' + APPLIANCE + '.csv')
