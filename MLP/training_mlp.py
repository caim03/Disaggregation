import warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000

from nilmtk import DataSet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mlpdisaggregator import MLPDisaggregator
from nilmtk.datastore import HDFDataStore
import sys
import pandas as pd

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

DATASET = '../data/ENEA/enea.h5'
MODEL = '../data/ENEA/model-mlp-' + APPLIANCE + 'enea.h5'
DISAG = '../data/ENEA/disag-mlp-' + APPLIANCE + 'out.h5'
#UKDALE_MODEL = '../data/UKDALE/dae-fridge-ukdale.h5' # Vale solo per il frigorifero per ora
TRAIN_BUILDING = 1
TEST_BUILDING = 1
SEQUENCE = 256

train = DataSet(DATASET)
train.set_window(start="2017-03-11", end="2017-09-30") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters

mlp = MLPDisaggregator(SEQUENCE, FINE_TUNING)

if FINE_TUNING:
    print("------ FINE TUNING ------")
    mlp.fine_tuning(UKDALE_MODEL)

train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()[APPLIANCE] # The kettle meter that is used as a training target

if TRAINING:
    print("------ TRAINING ------")
    mlp.train(train_mains, train_meter, epochs=5, sample_period=1)
    mlp.export_model(MODEL)
else:
    print("------ IMPORT MODEL ------")
    mlp.import_model(MODEL)

# dae.import_model("../data/UKDALE/dae-ukdale.h5")
test = DataSet(DATASET)
test.set_window(start="2017-10-01", end="2017-10-31")
test_elec = test.buildings[TEST_BUILDING].elec
test_mains = test_elec.mains()

from nilmtk.datastore import HDFDataStore
output = HDFDataStore(DISAG, 'w')

print("------ TESTING ------")
mlp.disaggregate(test_mains, output, train_meter, sample_period=1)

result = DataSet(DISAG)
res_elec = result.buildings[TEST_BUILDING].elec
predicted = res_elec[APPLIANCE]
ground_truth = test_elec[APPLIANCE]

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(ground_truth.power_series_all_data(), label='ground truth')
ax.plot(predicted.power_series_all_data(), label='predicted')
#plt.xlim('2017-10-08 00:00:00', '2017-10-08 01:00:00')
#plt.ylim(0, 300)
plt.xlabel('Time')
plt.ylabel('Power [W]')
plt.title(APPLIANCE + ' Disaggregation')
myFmt = mdates.DateFormatter('%d:%H:%M')
ax.xaxis.set_major_formatter(myFmt)
ax.legend()
plt.savefig(APPLIANCE + "_mlp.png")


import metrics
print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(predicted, ground_truth)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(predicted, ground_truth)))
print("============ List of percentages for every days\n")
date_series = pd.date_range(start="2017-10-01", end="2017-10-31", freq='D')

print(metrics.daily_relative_consume(predicted, ground_truth, test_mains, date_series))
