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
ERRBAR_FACTOR = 0.01  # factor for error bars if not present in the SED and filter files
ERRBAR_SAMPLES = 1000  # number of samples to draw when computing error bars


def filterfromfile(file, errbar=False):
    """
    Create a filter model from a file.
    """
    with open(filterpath + file + ".res", "r") as f:
        filter = np.loadtxt(f)

    if errbar:
        if filter.shape[1] == 3:
            return splrep(filter[:, 0], filter[:, 1], k=1, s=0), splrep(filter[:, 0], filter[:, 2], k=1, s=0)
        else:
            return splrep(filter[:, 0], filter[:, 1], k=1, s=0), splrep(
                filter[:, 0], filter[:, 1] * ERRBAR_FACTOR, k=1, s=0
            )
    else:
        return splrep(filter[:, 0], filter[:, 1], k=1, s=0)


def get_sed(name, errbar=False):
    """
    Returns a model of the SED, a tuple of (wave,data) or (wave,data,err)
    """
    with open(SEDpath + name + ".sed", "r") as f:
        sed = np.loadtxt(f)

    if errbar:
        if sed.shape[1] == 3:
            return sed[:, 0], sed[:, 1], sed[:, 2]
        else:
            return sed[:, 0], sed[:, 1], np.full_like(sed[:, 1], ERRBAR_FACTOR)
    else:
        return sed[:, 0], sed[:, 1]


def ab_filter_magnitude(filter, spectrum, redshift, errbar=False):
    """
    Determines the AB magnitude (up to a constant) given an input filter, SED,
        and redshift. Does not include cosmological dimming, so should only
        be used for relative magnitudes.
    """
    sol = 299792452.0  # speed of light in vacuum in m/s

    wave = spectrum[0].copy()
    data = spectrum[1].copy()
    if not errbar:
        samples = 1
    else:
        samples = ERRBAR_SAMPLES
        err = spectrum[2].copy()
        loc = np.array(data)[:, None]
        scale = np.array(err)[:, None]
        data_sampled = np.random.normal(loc, scale, size=(len(data), samples))
        filter_orig = filter  # tuples are immutable so no need to copy
        filter = filter[0]

    # Redshift the spectrum and determine the valid range of wavelengths
    wave *= 1.0 + redshift
    wmin, wmax = filter[0][0], filter[0][-1]
    cond = (wave >= wmin) & (wave <= wmax)

    if not errbar:
        # Evaluate the filter at the wavelengths of the spectrum
        response = splev(wave[cond], filter)
    else:
        response = splev(wave[cond], filter_orig[0])
        response_err = splev(wave[cond], filter_orig[1])
        loc = np.array(response)[:, None]
        scale = np.array(response_err)[:, None]
        response_sampled = np.random.normal(loc, scale, size=(len(response), samples))

    estimates = []
    for i in range(samples):

        data = data_sampled[:, i] if errbar else data
        response = response_sampled[:, i] if errbar else response

        # Convert to f_nu
        data = data * wave**2 / (sol * 1e10)

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

        result = -2.5 * log10(flux / bandpass) - 48.6
        estimates.append(result)

    return (np.mean(estimates), np.std(estimates)) if errbar else (estimates[0], 0)


def vega_filter_magnitude(filter, spectrum, redshift, errbar=False):
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
    vwave, vdata = get_sed("Vega", errbar=errbar)
    cond = (vwave >= wmin) & (vwave <= wmax)
    response = splev(vwave[cond], filter)
    vega = splrep(vwave[cond], response * vdata[cond], s=0, k=1)
    vegacorr = splint(wmin, wmax, vega)

    return -2.5 * log10(flux / vegacorr)  # +2.5*log10(1.+redshift)


def filter_magnitude(filter, spectrum, redshift, zp, errbar=False):

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
