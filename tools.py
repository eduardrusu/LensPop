"""
A module of tools to work with SED, spectral, and filter operations.
"""

import glob
import os
from math import log10, pi

import numpy as np
from scipy.interpolate import splev, splint, splrep

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filterpath = os.path.join(BASE_DIR, "filters/")
SEDpath = os.path.join(BASE_DIR, "templates/")

filter_list = glob.glob(filterpath + "*res")
filter_list.sort()
filter_list = [filter.split("/")[-1].split(".res")[0] for filter in filter_list]
sed_list = glob.glob(SEDpath + "*sed")
sed_list.sort()
sed_list = [sed.split("/")[-1].split(".sed")[0] for sed in sed_list]

BC03factor = 3.826e33 / (4 * pi * 3.08568e19**2)


def filterfromfile(file):
    """
    Create a filter model from a file.
    """
    with open(filterpath + file + ".res", "r") as f:
        filter = np.loadtxt(f)

    return splrep(filter[:, 0], filter[:, 1], k=1, s=0)


def get_sed(name):
    """
    Returns a model of the SED, a tuple of (wave,data)
    """
    with open(SEDpath + name + ".sed", "r") as f:
        sed = np.loadtxt(f)

    return sed[:, 0], sed[:, 1]


def ab_filter_magnitude(filter, spectrum, redshift):
    """
    Determines the AB magnitude (up to a constant) given an input filter, SED,
        and redshift. Does not include cosmological dimming, so should only
        be used for relative magnitudes.
    """
    sol = 299792452.0  # speed of light in vacuum in m/s

    wave = spectrum[0].copy()
    data = spectrum[1].copy()

    # Convert to f_nu
    data = data * wave**2 / (sol * 1e10)

    # Redshift the spectrum and determine the valid range of wavelengths
    wave *= 1.0 + redshift
    wmin, wmax = filter[0][0], filter[0][-1]
    cond = (wave >= wmin) & (wave <= wmax)

    # Evaluate the filter at the wavelengths of the spectrum
    response = splev(wave[cond], filter)

    freq = sol * 1e10 / wave[cond]
    data = data[cond] * (1.0 + redshift)

    # Flip arrays
    freq = freq[::-1]
    data = data[::-1]
    response = response[::-1]

    # Integrate
    observed = splrep(freq, response * data / freq, s=0, k=1)
    flux = splint(freq[0], freq[-1], observed)

    bp = splrep(freq, response / freq, s=0, k=1)
    bandpass = splint(freq[0], freq[-1], bp)

    return -2.5 * log10(flux / bandpass) - 48.6


def vega_filter_magnitude(filter, spectrum, redshift):
    """
    Determines the Vega magnitude (up to a constant) given an input filter,
        SED, and redshift.
    """

    wave = spectrum[0].copy()
    data = spectrum[1].copy()

    # Redshift the spectrum and determine the valid range of wavelengths
    wave *= 1.0 + redshift
    data /= 1.0 + redshift
    wmin, wmax = filter[0][0], filter[0][-1]
    cond = (wave >= wmin) & (wave <= wmax)

    # Evaluate the filter at the wavelengths of the spectrum
    response = splev(wave[cond], filter)

    # Determine the total observed flux (without the bandpass correction)
    observed = splrep(wave[cond], (response * data[cond]), s=0, k=1)
    flux = splint(wmin, wmax, observed)

    # Determine the magnitude of Vega through the filter
    vwave, vdata = get_sed("Vega")
    cond = (vwave >= wmin) & (vwave <= wmax)
    response = splev(vwave[cond], filter)
    vega = splrep(vwave[cond], response * vdata[cond], s=0, k=1)
    vegacorr = splint(wmin, wmax, vega)

    return -2.5 * log10(flux / vegacorr)  # +2.5*log10(1.+redshift)


def filter_magnitude(filter, spectrum, redshift, zp):

    wave = spectrum[0].copy()
    data = spectrum[1].copy()

    # Redshift the spectrum and determine the valid range of wavelengths
    wave *= 1.0 + redshift
    data /= 1.0 + redshift
    wmin, wmax = filter[0][0], filter[0][-1]
    cond = (wave >= wmin) & (wave <= wmax)

    # Evaluate the filter at the wavelengths of the spectrum
    response = splev(wave[cond], filter)

    # Determine the total observed flux (without the bandpass correction)
    observed = splrep(wave[cond], (response * data[cond]), s=0, k=1)
    flux = splint(wmin, wmax, observed)
    flux = (response * data[cond]).sum()

    return -2.5 * log10(flux) + zp


def save_lime_sed(input_path):
    """
    Given an SED of LIME coefficients for the Moon model, save the SED to a file
    readable by get_sed.
    """
    try:
        with open(input_path, "r") as f:
            lines = f.readlines()

        datetime = [
            line.split(",")[-1].strip("\n").replace(" ", "_").replace(":", "_")
            for line in lines
            if line.startswith("datetime")
        ][0]
        ind = [i for i, line in enumerate(lines) if line.startswith("Wavelengths (nm)")][0]
        lines = lines[ind + 1 :]
        wave = np.array([float(line.split(",")[0]) for line in lines]) * 10  # convert from nm to Angstroms
        irradiance = np.array([float(line.split(",")[1]) for line in lines])
        errors = np.array([float(line.split(",")[2]) for line in lines])
        data = np.vstack([wave, irradiance, errors]).T
        np.savetxt(SEDpath + f"moon_{datetime}.sed", data, header="Wavelength (Angstroms)  Irradiance  Error", fmt="%e")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

    return None
