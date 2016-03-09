#!/usr/bin/env python
# coding: utf-8

"""Various small mathematical functions

"""

import logging

import numpy
import numpy.linalg
import scipy
import scipy.optimize
import scipy.stats

from . import tools
from .meta import expanddoc
from . import ureg

inputs = """:param z: Height
    :type z: ndarray
    :param q: Quantity, dim 0 must be height.
    :type q: ndarray
    :param ignore_negative: Set negative values to 0
    :type ignore_negative: bool"""

@expanddoc
def layer2level(z, q, ignore_negative=False):
    """Converts layer to level. First dim. must be height.

    {inputs}
    :returns: Level-valued quantity.
    """
    dz = z[1:, ...] - z[:-1, ...]
    if ignore_negative:
        q[q<0]=0
    y_avg = (q[1:, ...] + q[:-1, ...])/2
    return (y_avg * numpy.atleast_2d(dz).T)

@expanddoc
def integrate_with_height(z, q, ignore_negative=False):
    """Calculate vertically integrated value

    {inputs}
    :returns: Vertically integrated value
    """

    return layer2level(z, q, ignore_negative).sum(0)

@expanddoc
def cum_integrate_with_height(z, q, ignore_negative=False):
    """Calculate cumulative integral with height

    {inputs}
    :returns: Cumulatively vertically integrated value
    """
    return layer2level(z, q, ignore_negative).cumsum(0)

#@tools.validator # comment out because fails for *args
def average_position_sphere(*args: (lambda a: len(a) in (1,2))):
    """Calculate the average position for a set of angles on a sphere

    This is quite imprecise, errors can be dozens of km.  For more
    advanced calculations, use proj4/pyproj.

    Input can be either:

    :param lat: Vector of latitudes
    :param lon: Vector of longitudes

    Or:

    :param locs: Nx2 ndarray with lats in column 0, lons in column 1
    """

    if len(args) == 1:
        locs = args[0]
        lat = locs[0, :]
        lons = locs[1, :]
    elif len(args) == 2:
        lat, lon = args

    X = numpy.cos(lat) * numpy.cos(lon)
    Y = numpy.cos(lat) * numpy.sin(lon)
    Z = numpy.sin(lat)

    xm = X.mean()
    ym = Y.mean()
    zm = Z.mean()

    lonm = numpy.arctan2(ym, xm)
    latm = numpy.arctan2(zm, numpy.sqrt(xm**2+ym**2))

    return (latm, lonm)

def linear_interpolation_matrix(x_old, x_new):
    """Get transformation matrix for linear interpolation.

    This is denoted by W in Calisesi, Soebijanta and Van Oss (2005).

    Does note extrapolate; values outside the range are equal to the
    outermost values.

    :param x_old: Original 1-D grid
    :param x_new: New 1-D grid for interpolation
    :returns ndarray W: Interpolation transformation matrix.
    """
    
    W = numpy.vstack(
        [numpy.interp(x_new, x_old, numpy.eye(x_old.size)[i, :],
            left=numpy.nan, right=numpy.nan)    
            for i in range(x_old.size)])
    #return W
    return W.T


#    return numpy.vstack(
#        [scipy.interpolate.InterpolatedUnivariateSpline(
#            x_old, eye(x_old.size)[i, :])(x_new) 
#                for i in range(x_old.size)])

def regrid_ak(A, z_old, z_new, cut=False):
    """Regrid averaging kernel matrix.

    Actual regridding done in apply_W_A, following Calisesi, Soebijanta
    and Van Oss (2005).

    :param A: Original averaging kernel
    :param z_old: Original z-grid
    :param z_new: New z-grid
    :param bool cut: Cut off, i.e. flag, when any z in the new grid is
        outside the old grid.
    :returns: (New averaging kernel, W)
    """

    if cut:
        # make sure we take care of flagged data
        valid = ~(A<-10).all(0)

        # Not on the new one!  We must output the same size for A every
        # time.
#        new_outside = ((z_new > numpy.nanmax(z_old)) | 
#                       (z_new < numpy.nanmin(z_old)))
        #               ~numpy.isfinite(z_old))
        z_old_valid = numpy.isfinite(z_old)
#        if z_old[z_old_valid].max() < z_new.max():
#            raise ValueError("z_new not a subset of z_old!")
        #W = linear_interpolation_matrix(z_old[z_old_valid], z_new[~new_outside])
        #W = linear_interpolation_matrix(z_old[z_old_valid], z_new)
        # Keep full W (unflagged) because I want to put them all in a
        # single ndarray later
        # TODO: retry with masked arrays after the bugfixes
        W = linear_interpolation_matrix(z_old, z_new)
        #z_new_ok = (z_new > z_old.min()) & (z_new < z_old.max())
        z_new_ok = numpy.isfinite(W).all(1)
        A_new = numpy.zeros(shape=(z_new.shape[0], z_new.shape[0]))
        A_new.fill(numpy.nan)
        # The first one seems to not work... the second one does
        #A_new[z_new_ok, :][:, z_new_ok] = apply_W_A(
        A_new[numpy.ix_(z_new_ok, z_new_ok)] = apply_W_A(
                W[:, z_old_valid][z_new_ok, :],
                A[z_old_valid, :][:, z_old_valid])
#        A_new[outside, outside] = numpy.nan
        return (A_new, W)
    else:
        W = linear_interpolation_matrix(z_old, z_new)
        return (apply_W_A(W, A), W)

def regrid_matrix(A, z_old, z_new):
    """Regrid single matrix between grids.

    Do not use for averaging kernels!

    :param A:
    :param z_old:
    :param nd-array z_new: 1-D array
    """

    if z_old.shape[1] != A.shape[1]:
        raise ValueError("Shapes dont match")
    scipy.interpolate.interp1d
    A_new = numpy.zeros(shape=(z_old.shape[0], z_new.shape[0]))
    for i in range(z_old.shape[0]):
        for x in range(A.shape[0]):
            ip = scipy.interpolate.interp1d(z_old[i, :], A[x, :],
                bounds_error=False)
            A_new[i, :] = ip(z_new)
    return A_new

def apply_W_A(W, A):
    """Regrid averaging kernel matrix using W

    If interpolation matrix W is already calculated, apply to averaging
    kernel matrix here.

    This follows the methodology outlined by Calisesi, Soebijanta and Van
    Oss (2005).
    """

    Wstar = numpy.linalg.pinv(W)
    return W.dot(A).dot(Wstar)

def convert_ak_ap2vmr(AKx, aprf):
    """Convert averaging kernel from SFIT4 units to vmr

    :param AKx: Averaging kernel from SFIT4
    :param aprf: A-priori
    :returns: Averaging kernel in VMR units
    """

    # Source: e-mail Stephanie 2014-06-17

    return numpy.diag(1/aprf).dot(AKx).dot(numpy.diag(aprf))

def smooth_profile(xh, ak, xa):
    """Calculated smoothed profile.

    Calculate a smoothed profile following Rodgers and Connor (2003).

    :param xh: High-resolution profile
    :param ak: Low-resolution averaging kernel [VMR]
    :param xa: Low-resolution a priori profile
    """

    OK = (numpy.isfinite(xa) &
          numpy.isfinite(numpy.diag(ak)) &
          numpy.isfinite(xh))
    xs = numpy.zeros_like(xa)
    xs.fill(numpy.nan)
    xs[OK] = xa[OK] + ak[numpy.ix_(OK,OK)].dot(xh[OK] - xa[OK])
    return xs

def mad(x):
    """Median absolute deviation
    """

    return numpy.median(numpy.abs(x - numpy.median(x)))

def get_transformation_matrix(f, n):
    """Obtain standard matrix for the linear transformaton

    For a given linear function taking a vector valued argument, obtain
    the standard matrix for the linear transformation.

    See Lay (2003), Linear Algebra and its Transformations, 3rd edition,
    Theorem 10 (page 83).

    :param callable f: Function for which to get the transformation
        matrix.  This might be a function or a functools.partial object,
        for example.  This function should take as input a vector of
        length n and return as output a vector of length m (m>0).
        Of course, this function should be linear.
    :param int n: Size of transformation matrix needed
    :returns: (m x n) transformation matrix.
    """

    I = numpy.eye(n)
    return numpy.hstack([f(I[:, i:(i+1)]) for i in range(n)])

def calc_bts_for_srf_shift(x, bt_master, srf_master,
                            y_spectral_db, f_spectra, L_spectral_db,
                            unit=ureg.um):
    """Calculate BTs estimating bt_target from bt_master assuming srf_master shifts by x

    Try to estimate bt_target from bt_master, assuming that bt_master are
    radiances corresponding to spectral response function srf_master.
    For the estimate, use a database described by y_spectral_db and
    f_spectra.  This database will be used to train a regression from
    the original bt (bt_master, due to srf_master) to a shifted bt (due to
    srf_master shited by x).  The regression is trained using the spectral
    database (y_spectral_db, f_spectra, L_spectral_db).  This function
    then applies the regression to bt_master.

    This function is designed to be called by
    `:func:calc_rmse_for_srf_shift`, which in turn is designed
    to be called repeatedly within an optimisation framework (see
    `:func:estimate_srf_shift`).  Therefore, as much as possible is
    precalculated before calling this.  Hence, you also need to pass
    L_ref, which equals integrated radiances for the reference satellite,
    corresponding to f_spectra and y_spectra.

    This estimate considers one channel at a time.  It may be more optimal
    to estimate multiple channels at a time.  This is to be done.

    Arguments:
        
        x (Quantity or float): shift in SRF.  Will be converted to the
            unit (typically µm or nm) from the pint user registry (see
            later argument).
        bt_master (ndarray): Radiances for reference satellite, in brightness
            temperatures [K].
        srf_master (`:func:pyatmlab.physics.SRF`): SRF for reference satellite
        y_spectral_db (ndarray N×p): Database of spectra (such as from IASI)
            to use.  Should be in spectral radiance per frequency units [W
            / (m^2 sr Hz)]
        f_spectra (ndarray N): frequencies corresponding to y_spectral_db [Hz]
        L_spectral_db (ndarray N×p: For the database of spectra, radiances
            (brightness temperatures) corresponding to srf_master.  This
            is equal to:

                srf_master.channel_radiance2bt(
                    srf_master.integrate_radiances(f_spectra, y_spectral_db))

            but should be pre-calculated because it is an expensive
            calculation and this function needs to be called many times.
        unit (Unit): unit from pint unit registry.  Defaults to ureg.um.


    Returns:

        ndarray with estimates for shifted bt1 values
    """
    try:
        x = x.to(unit)
    except AttributeError:
        x = x * unit
    srf_sh = srf_master.shift(x)
    L_target = srf_sh.channel_radiance2bt(
            srf_sh.integrate_radiances(f_spectra, y_spectral_db))

    (slope, intercept, r_value, p_value, stderr) = scipy.stats.linregress(
            L_spectral_db, L_other)
    
    return intercept*L_ref.u + slope*bt_master

def calc_rmse_for_srf_shift(x, bt_master, bt_target, srf_master,
                            y_spectral_db, f_spectra, L_spectral_db,
                            unit=ureg.um):
    """Calculate RMSE estimating bt_target from bt_master assuming srf_master shifts by x

    Try to estimate how well we can estimate bt_target from bt_master,
    assuming that bt_master are radiances
    corresponding to spectral response function srf_master.  For the estimate,
    use a database described by y_spectral_db and f_spectra.

    This function is designed to be called repeatedly within an
    optimisation framework (see `:func:estimate_srf_shift`).  Therefore,
    as much as possible is precalculated before calling this.  Hence, you
    also need to pass L_spectral_db, which equals integrated radiances for the
    reference satellite, corresponding to f_spectra and y_spectral_db.

    This estimate considers one channel at a time.  It may be more optimal
    to estimate multiple channels at a time.  This is to be done.

    For example, bt_master could be radiances corresponding to channel 1
    on MetOp-A, srf_master the SRF for channel 1 on MetOp-A.  bt_target
    could be radiances for NOAA-19, suspected to be approximated by
    shifting srf_master by an unknown amount.  x would be a shift amount
    to attempt, such as +30 nm.  We could then continue tring different
    values for x until we minimise the RMSE.

    FIXME: We should be shifting from an a priori NOAA-19 SRF, not from an
    a priori MetOp-A SRF.

    Arguments:
        
        x (Quantity): shift in SRF.
        bt_master (ndarray): Radiances for reference satellite, in brightness
            temperatures [K].
        bt_target (ndarray): Radiances for other satellite, in brightness
            temperatures [K].
        srf_master (`:func:pyatmlab.physics.SRF`): SRF for reference satellite
        y_spectral_db (ndarray N×p): Database of spectra (such as from IASI)
            to use.  Should be in spectral radiance per frequency units [W
            / (m^2 sr Hz)]
        f_spectra (ndarray N): frequencies corresponding to y_spectral_db [Hz]
        L_spectral_db: Radiances corresponding to y_spectral_db and f_spectra [K]
        unit (Unit): unit from pint unit registry.  Defaults to ureg.um.

    Returns:
        
        rmse (float): Root mean square error between brightness
            temperatures estimated by taking bt_master (which should correspond
            to srf_master) shifted by x, minus bt_target.
    """
    bt_estimate = calc_bts_for_srf_shift(x, bt_master, srf_master,
            y_spectral_db, f_spectra, L_spectral_db, unit)
    rmse = numpy.sqrt(((bt_target - bt_estimate)**2).mean())
    return rmse

def estimate_srf_shift(bt1, bt2, srf, y_spectra, f_spectra,
        **solver_args):
    """Estimate shift in SRF from pairs of brightness temperatures

    From pairs of brightness temperatures, estimate what SRF shifts
    minimises observed BT differences.

    Arguments:
        
        bt1 (ndarray): Radiances for reference satellite
        bt2 (ndarray): Radiances for other satellite
        srf (`:func:pyatmlab.physics.SRF`): SRF for reference satellite
        y_spectra (ndarray N×p): Database of spectra (such as from IASI)
            to use.  Should be in spectral radiance per frequency units.
        f_spectra (ndarray N): spectrum describing frequencies
            corresponding to `y_spectra`.  In Hz.
        **solver_args: Remaining arguments passed on to
            :func:`scipy.optimize.minimize_scalar`.  In particular, `args`
            must be a 1-tuple with the pint Unit object corresponding to
            the unit for the shift, for example, (ureg.um,).
    Returns:

        float: shift in SRF
    """

    # Use spectral database to derive regression bt1p = a + b * bt1, where
    # btp1 corresponds to a shift of x

    # then find x that minimises differences

    L_ref = srf.channel_radiance2bt(srf.integrate_radiances(f_spectra, y_spectra))

    def fun(x, unit=ureg.um):
        rmse = calc_rmse_for_srf_shift(x,
            bt1=bt1, bt2=bt2, srf=srf, y_spectra=y_spectra, f_spectra=f_spectra,
            L_ref=L_ref, unit=unit)
        logging.debug("Shifting {:9.4~}: rmse {:6.3f} K".format(
            x*unit, rmse))
        return rmse

    res = scipy.optimize.minimize_scalar(
            fun=fun,
            **solver_args)
#            bracket=[-0.1, 0.1],
#            bounds=[-1, 1],
#            method="brent",
#            args=(ureg.um,))
    return res
