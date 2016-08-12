#!/usr/bin/env python
# coding: utf-8

"""Various small mathematical functions

"""

import logging

import numpy
import numpy.ma
import numpy.linalg
import scipy
import scipy.optimize
import scipy.stats
import scipy.odr

import sklearn.linear_model
import pint

from . import tools
from .meta import expanddoc
from .units import ureg

inputs = """:param z: Height
    :type z: ndarray
    :param q: Quantity, dim 0 must be height.
    :type q: ndarray
    :param ignore_negative: Set negative values to 0
    :type ignore_negative: bool"""

# Make my own exception because scipy ODR doesn't raise, just gives a
# message in a list…
class ODRFitError(ArithmeticError):
    pass

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

def mad(x, masked=False):
    """Median absolute deviation
    """

    med = numpy.ma.median if masked else numpy.median

    return med(numpy.abs(x - med(x)))

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

def calc_y_for_srf_shift(Δλ, y_master, srf0, L_spectral_db, f_spectra, y_ref,
                           unit=ureg.um,
                           regression_type=sklearn.linear_model.LinearRegression,
                           regression_args={"fit_intercept": True},
                           predict_quantity="bt",
                           u_y_ref=None,
                           u_y_target=None):
    """Calculate radiances or BTs estimating y_target from y_master assuming srf0 shifts by Δλ

    Try to estimate y_target from y_master, assuming that y_master are
    bts or radiances corresponding to spectral response function srf0.
    For the estimate, use a database described by L_spectral_db and
    f_spectra.  This database will be used to train a regression from
    the original bt (y_master, due to srf0) to a shifted bt (due to
    srf0 shifted by Δλ).  The regression is trained using the spectral
    database (L_spectral_db, f_spectra, bt_ref).  This function
    then applies the regression to y_master.

    This function is designed to be called by
    `:func:calc_cost_for_srf_shift`, which in turn is designed
    to be called repeatedly within an optimisation framework (see
    `:func:estimate_srf_shift`).  Therefore, as much as possible is
    precalculated before calling this.  Hence, you also need to pass
    bt_ref, which equals integrated brightness temperatures for
    the reference satellite, corresponding to f_spectra and L_spectral_db.

    This estimate considers one channel at a time.  It may be more optimal
    to estimate multiple channels at a time.  This is to be done.

    Arguments:
        
        Δλ (Quantity or float): shift in SRF.  Will be converted to the
            unit (typically µm or nm) from the pint user registry (see
            later argument).  Scalar.
        y_master (Quantity ndarray, N×k): Brightness temperatures [K] or
            radiances [radiance units] for reference satellite.  N
            samples, k channels.  Quantity must be consistent with the one
            described by predict_quantity.  These are used in the actual
            prediction; the training is performed with y_ref.  For
            example, if we are predicting NOAA18-8 from NOAA19-1-12 in a
            test setting, y_master would correspond to NOAA19-1-12 test
            data, and y_ref would be the same for training data.  In the
            real world, y_ref is still simulated, but y_master are actual
            measurements.
        srf0 (`:func:pyatmlab.physics.SRF`): SRF relative to which
            the shift is to be calculated.
        L_spectral_db (ndarray M×l): Database of spectra (such as from IASI)
            to use.  Should be in spectral radiance per frequency units [W
            / (m^2 sr Hz)].  M spectra with l radiances each.
        f_spectra (Quantity ndarray l): frequencies corresponding to
            L_spectral_db [Hz].  1-D with length l.
        y_ref (Quantity ndarray M×k): Predictands used to train regressions,
            i.e. the training database.  This information follows directly from
            L_spectral_db and SRFs on the reference satellite, but it is
            an expensive calculation so should be pre-calculated.  If
            predicting from the same satellite, at least one channel will
            correspond to srf0, such that

                # in case of radiances
                L = srf0.integrate_radiances(f_spectra, L_spectral_db)
                # in case of BTs
                bt = srf0.channel_radiance2bt(L)

            but this is not the case if one satellite is predicted from
            another.
        unit (Unit): unit from pint unit registry.  Defaults to ureg.um.
        regression_type (scikit-learn regressor): Type of regression.
            Defaults to sklearn.linear_model.LinearRegression.  Other good
            option would be sklearn.cross_decomposition.PLSRegression.
            As long as regression_type(**regression_args) behaves like
            those two (with .fit and .predict), it should be OK.
        regression_args (dict): Keyword arguments to pass on to regressor.
            For example, for sklearn.linear_model.LinearRegression you
            would want to at least pass `{"fit_intercept": True}`.  For
            sklearn.cross_decomposition.PLSRegression you might use
            `{"n_components": 9, "scale": False}`.  Please refer to
            scikit-learn documentation.


    Returns:

        ndarray with estimates for shifted y_master or y_master values
    """
    try:
        Δλ = Δλ.to(unit)
    except AttributeError:
        Δλ = ureg.Quantity(Δλ, unit)
    srf_sh = srf0.shift(Δλ)
    L_target = srf_sh.integrate_radiances(f_spectra, L_spectral_db)
    if predict_quantity == "bt":
        y_target = srf_sh.channel_radiance2bt(L_target)
    elif predict_quantity == "radiance":
        y_target = L_target
    else:
        raise ValueError("Invalid predict_quantity: {:s}".format(predict_quantity))

    # sklearn wants 2-D inputs even when there is only a single predictand
    # atleast_2d makes it (1, N), I need (N, 1), but transposing the
    # result of atleast_2d would be wrong if the array was already 2D to
    # begin with
    if y_master.ndim == 1:
        y_master = y_master[:, numpy.newaxis]

    if y_ref.ndim == 1:
        y_ref = y_ref[:, numpy.newaxis]

    #clf = sklearn.linear_model.LinearRegression(fit_intercept=True)
    if issubclass(regression_type, sklearn.base.RegressorMixin):
        clf = regression_type(**regression_args)
        clf.fit(y_ref.m, y_target.m)
        return ureg.Quantity(clf.predict(y_master.m).squeeze(), y_master.u)
    elif issubclass(regression_type, scipy.odr.odrpack.ODR):
        try:
            sx = u_y_ref.to(y_ref.u, "radiance").m.squeeze()
            sy = u_y_target.to(u_y_target.u, "radiance").m.squeeze()
        except pint.DimensionalityError: # probably noise in other unit
            # here, those are really only used as weights, so same-unit
            # is not essential?  Otherwise I need to convert them, a
            # non-trivial step.
            sx = u_y_ref.m.squeeze()
            sy = u_y_target.m.squeeze()

        # meaningful initial guess: new value equal to old channel, rest 0
        # β0 has a offset and then one slope for each channel
        # don't set to zero, says odrpack guide section 1.E.
        β0 = numpy.zeros(shape=(y_ref.shape[1]+1,))+0.001
        samech = abs((y_target[:, numpy.newaxis] - y_ref)).mean(0).argmin()
        β0[samech+1] = 1
        if (sx==0).any() or (sy==0).any():
            mydata = scipy.odr.RealData(y_ref.m.T, y_target.m)
        else:
            mydata = scipy.odr.RealData(y_ref.m.T, y_target.m, sx=sx, sy=sy)
        myodr = scipy.odr.ODR(mydata, scipy.odr.multilinear, beta0=β0)
        myout = myodr.run()
        if not any(x in myout.stopreason for x in
            {"Sum of squares convergence",
             "Iteration limit reached",
             "Parameter convergence",
             "Both sum of squares and parameter convergence"}):
            raise ODRFitError("ODR fitting did not converge.  "
                "Stop reason: {:s}".format(myout.stopreason[0]))
        return ureg.Quantity(myout.beta[0], y_master.u) + (
            myout.beta[numpy.newaxis, 1:] * y_master).sum(1)
    else:
        raise ValueError("Unknown type of regression!")
#    (slope, intercept, r_value, p_value, stderr) = scipy.stats.linregress(
#            bt_ref, bt_target)
#    
#    return intercept*bt_ref.u + slope*bt_master

def calc_cost_for_srf_shift(Δλ, y_master, y_target, srf0,
                            L_spectral_db, f_spectra, y_ref,
                            unit=ureg.um,
                            regression_type=sklearn.linear_model.LinearRegression,
                            regression_args={"fit_intercept": True},
                            cost_mode="total",
                            predict_quantity="bt",
                            u_y_ref=None,
                            u_y_target=None):
    """Calculate cost function estimating y_target from y_master assuming
    srf0 shifts by Δλ

    Try to estimate how well we can estimate y_target from y_master,
    assuming that y_master are radiances or BTs
    corresponding to spectral response function srf0.  For the estimate,
    use a database described by L_spectral_db and f_spectra.

    This function is designed to be called repeatedly within an
    optimisation framework (see `:func:estimate_srf_shift`).  Therefore,
    as much as possible is precalculated before calling this.  Hence, you
    also need to pass y_ref, which equals integrated radiances for the
    reference satellite, corresponding to f_spectra and L_spectral_db,
    even though y_ref is redundant information.

    This estimate considers one channel at a time.  It may be more optimal
    to estimate multiple channels at a time.  This is to be done.

    For example, y_master could be radiances/BTs corresponding to channel 1
    on MetOp-A, srf0 the SRF for channel 1 on MetOp-A.  y_target
    could be radiances or BTs for NOAA-19, suspected to be approximated by
    shifting srf0 by an unknown amount.  Δλ would be a shift amount
    to attempt, such as +30 nm.  We could then continue trying different
    values for Δλ until we minimise the cost.

    The two alternative cost functions are:

        C₁ = \sum_{i=1}^N (y_est,i - y_ref,i)^2

    and

        C₂ = \sum_{i=1}^N (y_est,i - y_ref,i - <y_est,i - y_ref,i>)^2

    Arguments:
        
        Δλ (Quantity): shift in SRF.
        y_master (ndarray): Radiances [radiance units] or BTs [K] for
            reference satellite.  This comes from either the testing data,
            or actual measurements.  Quantity must be consistent with what is
            contained in predict_quantity.
        y_target (ndarray): Radiances or BTs for other satellite.  This
            comes from either the testing data (calculated by shifting
            srf0 by an amount you haven't told me, but I mean to recover),
            or from actual measurements.
        srf0 (`:func:pyatmlab.physics.SRF`): SRF corresponding to
            zero shift, relative to which the shift is estimated.
        L_spectral_db (ndarray N×p): Database of spectra (such as from IASI)
            to use.  Should be in SI spectral radiance per frequency units [W
            / (m^2 sr Hz)], regardless of predict_quantity.
        f_spectra (ndarray N): frequencies corresponding to L_spectral_db [Hz]
        y_ref: Radiances or brightness temperatures corresponding to 
            L_spectral_db and f_spectra [K], for all channels to be used.
            This is essential the training database.  It comes from
            simulations regardless as to we are applying this as a test or
            in the real world.
        unit (Unit): unit from pint unit registry.  Defaults to ureg.um.
        regression_type (scikit-learn regressor): Type of regression.
            Defaults to sklearn.linear_model.LinearRegression.  Other good
            option would be sklearn.cross_decomposition.PLSRegression.
            As long as regression_type(**regression_args) behaves like
            those two (with .fit and .predict), it should be OK.  Some
            other cases are also understood, most notably scipy.odr.ODR.
        regression_args (dict): Keyword arguments to pass on to regressor.
            For example, for sklearn.linear_model.LinearRegression you
            would want to at least pass `{"fit_intercept": True}`.  For
            sklearn.cross_decomposition.PLSRegression you might use
            `{"n_components": 9, "scale": False}`.  Please refer to
            scikit-learn documentation for details.
        cost_mode (str): How to estimate the cost.  Can be "total"
            (default), which calculates C₁, or "anomalies", which
            calculates C₂, according to their definitions above.
        predict_quantity (str): "bt" (default) or "radiance"
        u_y_ref (ndarray): Uncertainties on y_ref.  Used by some
            regression types.
        u_y_target (ndarray): Uncertainties on y_target.

    Returns:
        
        cost (float): Value of estimated cost function.  Unitless.
    """
    y_estimate = calc_y_for_srf_shift(Δλ, y_master, srf0,
            L_spectral_db, f_spectra, y_ref, unit,
            regression_type=regression_type,
            regression_args=regression_args,
            predict_quantity=predict_quantity,
            u_y_ref=u_y_ref,
            u_y_target=u_y_target)
    diffs = y_target - y_estimate
    if cost_mode.lower() == "anomalies":
        diffs -= (y_target - y_estimate).mean()
    elif cost_mode.lower() != "total":
        raise ValueError("Invalid cost mode: {:s}.  I understand 'total' "
                         "and 'anomalies'".format(cost_mode))
    cost = (diffs**2).sum() / (diffs.size * y_target.mean()**2)
    return cost

def estimate_srf_shift(y_master, y_target, srf0, L_spectral_db, f_spectra,
        y_ref,
        regression_type, regression_args,
        optimiser_func, optimiser_args,
        cost_mode,
        predict_quantity="bt",
        u_y_ref=None,
        u_y_target=None,
        **solver_args):
    """Estimate shift in SRF from pairs of brightness temperatures

    From pairs of brightness temperatures, estimate what SRF shifts
    minimises observed BT differences.

    Arguments:
        
        y_master (ndarray): Radiances or BTs for reference satellite.
            Unit must be consistent with what you tell me in
            predict_quantity.
        y_target (ndarray): Radiances or BTs for other satellite
        srf0 (`:func:pyatmlab.physics.SRF`): SRF for reference satellite
        L_spectral_db (ndarray N×p): Database of spectra (such as from IASI)
            to use.  Should ALWAYS be in spectral radiance per frequency
            units, regardless of what predict_quantity is.
        f_spectra (ndarray N): spectrum describing frequencies
            corresponding to `L_spectral_db`.  In Hz.
        y_ref: Reference BT or radiance
        regression_type (scikit-learn regressor): As for
            `:func:calc_cost_for_srf_shift`.
        regression_args (scikit-learn regressor): As for
            `:func:calc_cost_for_srf_shift`.
        optimiser_func (function): Function implementing optimising.
            Should take as a first argument a function to be optimised,
            remaining arguments taken from optimiser_args.
            Should return an instance of
            `:func:scipy.optimize.optimize.OptimizeResult`.  You probably
            want to select a function from `:module:scipy.optimize`, such
            as `:func:scipy.optimize.basinhopping` when there may be
            multiple minima, or `:func:scipy.optimize,minimize_scalar`
            when you expect only one.
        optimiser_args (dict): Keyword arguments to pass on to
            `optimiser_func`.
        cost_mode (str): As for `:func_calc_cost_for_srf_shift`.
        predict_quantity (str): Whether to perform prediction in
            "radiance" or in "bt".  y_master and y_target should be in
            units corresponding to this quantity.
        u_y_ref (ndarray):
        u_y_target (ndarray):
    Returns:

        float: shift in SRF
    """

    # Use spectral database to derive regression yp1 = a + b * y_master, where
    # yp1 corresponds to a shift of Δλ (radiance or BT)

    # then find Δλ that minimises differences

    #bt_ref = srf_master.channel_radiance2bt(srf_master.integrate_radiances(f_spectra, L_spectral_db))

    def fun(Δλ, unit=ureg.um):
        if isinstance(Δλ, numpy.ndarray) and Δλ.ndim > 0:
            if Δλ.size != 1:
                raise ValueError("Δλ must be scalar.  Found "
                    "Δλ.size=={:d}".format(Δλ.size))
            Δλ = Δλ.ravel()[0]
        cost = calc_cost_for_srf_shift(Δλ,
            y_master=y_master, y_target=y_target, srf0=srf0, L_spectral_db=L_spectral_db,
            f_spectra=f_spectra, y_ref=y_ref, unit=unit,
            regression_type=regression_type,
            regression_args=regression_args,
            cost_mode=cost_mode,
            predict_quantity=predict_quantity,
            u_y_ref=u_y_ref,
            u_y_target=u_y_target)
        logging.debug("Shifting {:0<13.7~}: cost {:0<12.8~}".format(
            Δλ*unit, cost))
        return cost.to("1").m

    res = optimiser_func(fun, **optimiser_args)

#    res = scipy.optimize.minimize_scalar(
#            fun=fun,
#            **solver_args)
#            bracket=[-0.1, 0.1],
#            bounds=[-1, 1],
#            method="brent",
#            args=(ureg.um,))
    return res

def filter_outliers(x, cut=5.0):
    """Empirical removal of outliers

    Masks values that are more than *n* times the MAD from the median.

    Arguments:
        
        x (ndarray)
            Array to process

        cut (float)
            Cutoff beyond which to mask.  Defaults to 5.0.
    """

    norm_dist = abs(x - numpy.ma.median(x))/mad(x, masked=True)
    x.mask[norm_dist>=cut] = True

    return x
