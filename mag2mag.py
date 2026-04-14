import argparse
from math import log10

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev

import distances
import tools

USAGE = """
NAME
        mag2mag

PURPOSE
        Given a galaxy of type T at redshift z1 with magnitude m1 in
        filter1, compute magnitude m2 in filter2 for the same galaxy at
        redshift z2.

COMMENTS
        Tricks the hyper-z photometric redshift code into doing this.
        PGPLOT module requires /usr/local/bin/perl at KIPAC.

USAGE
        mag2mag [flags] [options]

FLAGS
        -u           Print this message [0]
        -q           quiet operation [0]
        -vega        Input magnitudes are in Vega system [def=AB]
        -convert     Convert from one magnitude system to the other [def=0]
        -H0          Hubble constant [70]
        -Om          Omega matter [0.3]
        -OL          Omega lambda [0.7]
        -plot        Illustrate results with a nice plot
        -test        Do an example run (the first one on the list below)
        -errbar       Compute error bars

INPUTS
        -m1       f         input magnitude
        -m1_err   f         error on input magnitude (if -errbar is set)
        -f1       s         input filter
        -z1       f         input redshift
        -T        s         galaxy spectral type
        -f2       s         output filter
        -z2       f         output redshift

OPTIONAL INPUTS

OUTPUTS
        stdout       Useful information

EXAMPLES

        mag2mag -T CWW_Sbc_ext -m1 25 -f1 'Johnson_H'    -z1 0.6 \\
                 -plot -vega -convert -f2 'F814W_ACSWFC' -z2 1.4

        mag2mag -T CWW_E_ext -m1 -21.43 -f1 'Johnson_B' -z1 0.0 \\
                   -plot -vega -convert -f2 'F775W_ACSWFC' -z2 0.321

        mag2mag -T CWW_E_ext -m1 18.302625 -f1 'F775W_ACSWFC' -z1 0.321 \\
                            -plot -convert -f2 'Johnson_B'    -z2 0.0
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mag2mag",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=USAGE,
    )

    parser.add_argument("-u", action="store_true", dest="show_usage")
    parser.add_argument("-q", action="store_true", dest="quiet")
    parser.add_argument("-vega", action="store_true", dest="vega")
    parser.add_argument("-convert", action="store_true", dest="convert")
    parser.add_argument("-plot", action="store_true", dest="plot")
    parser.add_argument("-test", action="store_true", dest="test")
    parser.add_argument("-errbar", action="store_true", dest="errbar")

    parser.add_argument("-m1", type=float)
    parser.add_argument("-m1_err", type=float)
    parser.add_argument("-f1")
    parser.add_argument("-z1", type=float)
    parser.add_argument("-T", dest="sedtype")
    parser.add_argument("-f2")
    parser.add_argument("-z2", type=float)

    parser.add_argument("-H0", type=float, default=70.0)
    parser.add_argument("-Om", type=float, default=0.3)
    parser.add_argument("-OL", type=float, default=0.7)

    return parser


def mag2mag(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_usage:
        print(USAGE)
        raise SystemExit(0)

    if args.test:
        args.sedtype = "Sbc_cww"
        args.m1 = 25.0
        args.f1 = "H_Johnson"
        args.z1 = 0.6
        args.f2 = "F814W_WFC"
        args.z2 = 1.4
        args.plot = True
        args.vega = True
        args.convert = True

    if args.m1 is None or args.f1 is None or args.z1 is None or args.sedtype is None:
        parser.error("Incomplete input information. Use -u for usage.")

    if (args.m1_err is None and args.errbar) or not args.errbar:
        args.m1_err = 0

    f1 = tools.filterfromfile(args.f1, errbar=args.errbar)
    f2 = f1 if args.f2 is None else tools.filterfromfile(args.f2, errbar=args.errbar)
    z2 = args.z1 if args.z2 is None else args.z2

    sed = tools.get_sed(args.sedtype, errbar=args.errbar)

    if args.vega:
        filtermag, filtermag_err = tools.vega_filter_magnitude(f1, sed, args.z1, errbar=args.errbar)
    else:
        filtermag, filtermag_err = tools.ab_filter_magnitude(f1, sed, args.z1, errbar=args.errbar)

    magoffset = args.m1 - filtermag

    if args.vega:
        m2, m2_err = tools.vega_filter_magnitude(f2, sed, z2, errbar=args.errbar)
        m2 += magoffset
    else:
        m2, m2_err = tools.ab_filter_magnitude(f2, sed, z2, errbar=args.errbar)
        m2 += magoffset
    m2_err = np.sqrt(m2_err**2 + filtermag_err**2 + args.m1_err**2)

    if args.z1 != z2:
        dist = distances.Distance()
        dist.OMEGA_M = args.Om
        dist.OMEGA_L = args.OL
        dist.h = args.H0 / 100.0

        if z2 != 0.0:
            dimming = dist.Dl(args.z1) / dist.Dl(z2)
        else:
            dimming = dist.Dl(args.z1) / 1e-5

        m2 -= 5.0 * log10(dimming)

    if args.convert:
        vegatmp = tools.get_sed("Vega")
        vegatmp = (vegatmp[0], vegatmp[1])
        vega_ab = tools.ab_filter_magnitude(f2, vegatmp, 0.0)
        if args.vega:
            m2 += vega_ab
        else:
            m2 -= vega_ab

    if not args.quiet:
        print(f"{m2:.4f} +/- {m2_err:.4f}")

    if args.plot:
        plt.plot(sed[0] * (1.0 + args.z1), sed[1] / sed[1].mean(), c="k", label="Input SED")
        if args.z1 != z2:
            plt.plot(sed[0] * (1.0 + z2), sed[1] / sed[1].mean(), c="r")

        wave = sed[0] * (1.0 + args.z1)
        if args.errbar:
            f1 = f1[0]
            f2 = f2[0]
        cond = (wave >= f1[0][0]) & (wave <= f1[0][-1])
        plt.plot(wave[cond], splev(wave[cond], f1), c="b", label="Input Filter")

        same_filter = len(f1) == len(f2) and all(np.array_equal(a, b) for a, b in zip(f1, f2))
        if not same_filter:
            wave = sed[0] * (1.0 + z2)
            cond = (wave >= f2[0][0]) & (wave <= f2[0][-1])
            plt.plot(wave[cond], splev(wave[cond], f2), c="r", label="Output Filter")

        plt.xlabel("Wavelength (Angstroms)")
        plt.ylabel("Normalized Flux")
        plt.legend()
        plt.show()

    return (m2, m2_err)


if __name__ == "__main__":
    mag2mag()
