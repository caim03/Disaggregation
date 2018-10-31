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
DATASET = '../data/ENEA/enea.h5'
MODEL = '../data/ENEA/model-dae-enea.h5'
DISAG = '../data/ENEA/disag-dae-out.h5'
APPLIANCE = 'fridge'
TRAIN_BUILDING = 1
TEST_BUILDING = 1
SEQUENCE = 256

train = DataSet(DATASET)
train.set_window(start="2017-03-11", end="2017-08-31") # Training data time window
train_elec = train.buildings[TRAIN_BUILDING].elec # Get building 1 meters

dae = DAEDisaggregator(SEQUENCE)

train_mains = train_elec.mains() # The aggregated meter that provides the input
train_meter = train_elec.submeters()[APPLIANCE] # The kettle meter that is used as a training target

print("------ TRAINING ------")
dae.train(train_mains, train_meter, epochs=50, sample_period=1)
dae.export_model(MODEL)

# dae.import_model("../data/UKDALE/dae-ukdale.h5")
test = DataSet(DATASET)
test.set_window(start="2013-09-01", end="2013-10-31")
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
ax.plot(predicted.power_series_all_data(), label='predicted')
ax.plot(ground_truth.power_series_all_data(), label='ground truth')
#plt.xlim('2013-06-22 00:00:00', '2013-06-22 23:59:00')
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
