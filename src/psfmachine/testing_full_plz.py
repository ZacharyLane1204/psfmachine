import psfmachine as psf
import lightkurve as lk
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)

ra = 202.9625
dec = -10.736
folder = "/Users/zgl12/Research/K2_Files/wcs/"

files = sorted(os.listdir(folder))
files = [folder + f for f in files]

print(files)

# for i in range(len(files)):
#     with fits.open(files[i], mode='update') as hdul:
#         data = hdul[0].data  # Assuming the data is in the primary HDU

#         data[np.isnan(data)] = 0

#         if np.nanmedian(data) != 100:

#             data += 100

#         hdul.flush()

machine = psf.SSMachine.from_file(files, magnitude_limit=17, dr=3)

# machine.fit_lightcurves(plot=True, iter_negative=True, fit_mean_shape_model=False, fit_va=True, sap=True)

machine.build_frame_shape_model(plot=True)
machine.time_corrector = "centroid"
machine.n_time_points = 25
machine.build_time_model(plot=True)
plt.show()

machine.fit_frame_model()
machine.compute_aperture_photometry(aperture_size="optimal", target_complete=1, target_crowd=1)


plt.figure()
for i in range(500):
    data = machine.lcs[i].to_pandas().to_numpy()
    plt.plot(machine.time, machine.ws_frame[:, i])
# plt.plot(times, data1[:,0])
# plt.ylim(-500, 2000)
plt.show()