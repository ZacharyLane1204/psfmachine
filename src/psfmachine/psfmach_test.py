import psfmachine
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Angle
from astroquery.gaia import Gaia
import astropy.units as u
import time

def get_gaia_region(ra,dec,size=0.4, magnitude_limit = 15):
    """
    Get the coordinates and mag of all gaia sources in the field of view.
 
    -------
    Inputs-
    -------
    tpf class target pixel file lightkurve class
    magnitude_limit float cutoff for Gaia sources
    Offset int offset for the boundary
 
    --------
    Outputs-
    --------
    coords array coordinates of sources
    Gmag array Gmags of sources
    """
    c1 = SkyCoord(ra, dec, unit='deg')
    Vizier.ROW_LIMIT = -1
 
    result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                     radius=Angle(size, "arcmin"),column_filters={'Gmag':f'<{magnitude_limit}'})
 
    # keys = ['objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','gKmag','e_gKmag','rmag',
    #         'e_rmag','rKmag','e_rKmag','imag','e_imag','iKmag','e_iKmag','zmag','e_zmag','zKmag','e_zKmag',
    #         'ymag','e_ymag','yKmag','e_yKmag','tmag','gaiaid','gaiamag','gaiadist','gaiadist_u','gaiadist_l',
    #         'row','col', 'phot_g_mean_mag', 'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_g_mean_mag_error']
 
 
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message
 
    print(result)
    result = result['I/345/gaia2'].to_pandas()
    #result = result.rename(columns={'RA_ICRS':'ra','DE_ICRS':'dec'})
    #account for proper motion
    return result

def query_gaia_region(ra, dec, radius_deg, maglim = 17):
    # Define the central coordinates of your region
    # Gaia.MAIN_GAIA_TABLE.set_limit(10000)
    Gaia.ROW_LIMIT = 30000
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    # Define the search radius (half of the square side length)
    radius = radius_deg * u.degree

    # Query Gaia data
    job = Gaia.cone_search_async(coords, radius=radius)
    result = job.get_results().to_pandas()

    result = result[result['phot_g_mean_mag'] < maglim]

    return result

start = time.time()

folder = '/Users/zgl12/Research/K2_Files/'
filenames_og = 'k2mosaic-c06-ch61-cad112000_og.fits'
filenames_wcs = 'k2mosaic-c06-ch61-cad112000_wcs.fits'

hdu = fits.open(folder+filenames_og)

header = hdu[0].header
uncert = hdu[2].data
hdu.close()

data, hdr = fits.getdata(folder+filenames_wcs, header=True)

wcs = WCS(hdr)

jd = (header['MJD-BEG'] + header['MJD-END'])/2 + 2400000.5

data_shape = data.shape

y_indices, x_indices = np.indices(data_shape)
pixel_coords = np.column_stack([x_indices.flatten(), y_indices.flatten()])
world_coords = wcs.all_pix2world(pixel_coords, 0)

ras = world_coords[:, 0]
decs = world_coords[:, 1]

ras = ras.reshape(data_shape)
decs = decs.reshape(data_shape)

ra = 202.9625
dec = -10.736
radius = 5

# coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

# result = Gaia.cone_search_async(coord, radius*u.degree)
# cat = result.get_results().to_pandas()
 
# cat = get_gaia_region([ra],[dec],size=120)
cat = query_gaia_region(ra, dec, 2)

cat = cat[cat['phot_g_mean_mag'] < 17]
# print(len(cat))
 
print(cat.columns)

print(cat)

print(f"total time: {(time.time()-start)/60:.2f} minutes")

# ra_s = cat['RA_ICRS']
# dec_s = cat['DE_ICRS']
ra_s = cat['ra']
dec_s = cat['dec']

# Stack RA and DEC into a single array
world_coords_s = np.column_stack([ra_s, dec_s])

# Convert world coordinates to pixel coordinates
pixel_coords = wcs.all_world2pix(world_coords_s, 0)

# Extract x and y pixel coordinates
x_pix = pixel_coords[:, 0]
y_pix = pixel_coords[:, 1]

# plt.figure()
# plt.imshow(data)
# plt.scatter(x_pix, y_pix, s=5, c='r')
# plt.xlim(0, data_shape[1])
# plt.ylim(0, data_shape[0])
# plt.show()

# time: numpy.ndarray
#     Time values in JD
# flux: numpy.ndarray
#     Flux values at each pixels and times in units of electrons / sec
# flux_err: numpy.ndarray
#     Flux error values at each pixels and times in units of electrons / sec
# ra: numpy.ndarray
#     Right Ascension coordinate of each pixel
# dec: numpy.ndarray
#     Declination coordinate of each pixel
# sources: pandas.DataFrame
#     DataFrame with source present in the images
# column: np.ndarray
#     Data array containing the "columns" of the detector that each pixel is on.
# row: np.ndarray
#     Data array containing the "rows" of the detector that each pixel is on.

jds = [jd]

# cat['ra'] = cat['RA_ICRS']
# cat['dec'] = cat['DE_ICRS']

machine = psfmachine.Machine(time = jds, flux = data, flux_err = uncert, 
                             ra = ras, dec = decs, sources = cat, 
                             column= x_indices, row = y_indices, sources_flux_column = 'phot_g_mean_flux')

print(f"total time: {(time.time()-start)/60:.2f} minutes")

print(machine.sources)
print(machine.fit_lightcurves(plot=True))
print(machine.lcs[0])


