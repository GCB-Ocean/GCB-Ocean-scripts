"""
GCB evaluation script
=====================
This is a standalone script that can be used to calculate RMSE of SOCAT vs GCB fco2
The main function to run is `evaluate_sfco2`

See the docs in an interactive session (e.g. jupyter notebook)


Usage
-----
Copy this file to the same folder as the notebook you're running from then run
    >>> from evaluation_sfco2_socat import evaluate_sfco2
    >>> results_annual = evaluate_sfco2(sfco2_data, method='annual')
    >>> results_monthly = evaluate_sfco2(sfco2_data, method='monthly')
    
Note that this script will download SOCAT data automatically for comparison
You can set this caching folder in the evaluate_sfco2 function.

See the documentation of rmse_annual/rmse_monthly for more information 
on the difference between these two functions. 


GCB implementation
------------------
The GCB uses the annual comparison + IAV (on fgco2) but note that a mask is 
applied to the data prior to inputting to the evaluate_sfco2 function.
This mask is determined by the coverage of the pco2 products so that all 
products and models have equal coverage. 


Requirements
------------
Python >= 3.8

numpy
pooch
pandas
xarray

standard libraries not included in this list
`pip install xarray pooch` should install all dependencies


Information
-----------
author :   Luke Gregor
contact :  Luke.Gregor@usys.ethz.ch
           Judith.Hauck@awi.de
date :     2022-09-15
"""

import xarray as xr
import pandas as pd
import numpy as np
import pathlib
from functools import lru_cache, wraps
import logging

# constants that are set for this script. These can be changed after initialisation (see Demo)
SOCAT_URL = "https://socat.info/socat_files/v2023/SOCATv2023_tracks_gridded_monthly.nc.zip"
# will be used to create caching directories. 
# expects a folder in the base directory called ./data_cache
BASE = pathlib.Path(__file__).parent.expanduser().resolve().absolute()
YEAR_END = pd.Timestamp.today().year - 1  # the end year (e.g. set to 2022 for GCB-v2023)
LOG_LEVEL = 20  # sets the verbosity of output (higher is less verbose)



class Logger(logging.Logger):
    """
    A logger that is easier to set up than the default.
    Also provides more consistent behaviour with the
    logging levels as the handle and logger levels are
    set at the startup.
    Basically, syntactic sugar :)
    """

    def __init__(self, name="LOG", level=logging.INFO, format="[%name] %message"):
        """
        Returns a Python logger with more logical logging levels

        Parameters
        ----------
        name : str
            Name of the logger (will appear in front of each message)
        level : int
            logging level as defined in the logging module. Higher is quieter
            10 = debugging,
            20 = info,
            30 = warning,
        format : str
            Format of the messages. The default is "[%(name)s] %(message)s".
            The format can be customized by using the following placeholders:
            %name = name of the logger
            %message = message to log
            %<timeformat> = how to format the time (e.g. %Y-%m-%d OR %H:%M:%S)

        Returns
        -------
        logger : logging.Logger
            The logger object that has been adapted
        """

        import sys
        import io

        logging.Logger.__init__(self, name, level=level)

        format, date_format = self._get_logger_format(format)
        formatter = logging.Formatter(fmt=format, datefmt=date_format)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(self.level)
        self.addHandler(handler)

        self._output = io.StringIO()
        handler = logging.StreamHandler(self._output)
        handler.setFormatter(formatter)
        handler.setLevel(20)
        self.addHandler(handler)

        self.setLevel(self.level)

        self.log(15, "Logger initialized with level {}".format(self.level))

    @staticmethod
    def _get_logger_format(format):
        """
        Finds the date in the format string
        """
        import re

        if "%name" in format:
            format = format.replace("%name", "%(name)s")
        if "%message" in format:
            format = format.replace("%message", "%(message)s")

        pattern = "(%[YmdHMS]{1}[\s\./:_-]?(?=[^a-zA-Z]))"  # noqa
        matches = re.findall(pattern, format)
        date_format = "".join(matches)
        if date_format:
            format = format.replace(date_format, "%(asctime)s")
        else:
            date_format = None
        return format, date_format

    def log(self, level, msg, *args, **kwargs):
        """
        Logs a message with the given verbosity level.

        Parameters
        ----------
        level : int
            message loudness. Higher is louder
        """
        self._log(level, msg, args, **kwargs)

    @property
    def history(self):
        """
        Returns the history of the logger
        """
        return self._output.getvalue()


# create logger for logging when evaluate_sfco2 isn't run first 
# (which also creates a logger)
logger = Logger(f"GCB-{YEAR_END+1}", format="[%name: %Y-%m-%d %H:%M:%S] %message")
log_func = logger.log
log_func(20, f"Welcome to the Global Carbon Budget - Ocean CO2 evaluation script")
log_func(20, f"GCB-Ocean evaluation final year of analysis is {YEAR_END} (based on current year - 1)")
log_func(20, "To change, run gcb_eval.YEAR_END = <your chosen year>")
log_func(20, f"SOCAT will be downloaded from {SOCAT_URL}\n")


def evaluate_sfco2(sfco2, region='all', method='annual', cache_dir=f"{BASE}/data_cache/", log_level=LOG_LEVEL):
    """
    Evaluates predicted sfco2 against SOCATv202X sfCO2
    
    Note 
    ----
    You will have to apply a mask to the input data to ensure 
    that all models and pCO2-products are compared fairly since
    not all products have the same coverage. This can heavily 
    influence the results since marginal regions that are not 
    covered by all methods often have the largest variability 
    and thus uncertainty. 

    Parameters
    ----------
    sfco2 : [np.ndarray, xr.DataArray]
        Must be a 3-dimensional array [time, lat, lon] with 
        lat = [-90, 90], lon = [-180, 180]. Assumed that time ends
        on 202X-12-15 (set in global options)
    method : str, optional
        'monthly' or 'annual', by default 'annual'
        'annual' compares the spatially-averaged time series 
        'monthly' does a point-wise comparison and then averages spatially
    region : [int, str]
        choose from one of three regions. global=0, north=1, tropics=2, south=3, all=4
        can be a string or an integer value, by default 'all'
    cache_dir : path, optional
        The path to the cache where SOCAT and other data will be downloaded to and 
        stored, by default f"{BASE}/data_cache/"

    Returns
    -------
    xr.Dataset
        the evaluation statistics for sfCO2 (differs depending on the method)
        but will always contain RMSE and bias. Also contains the gridded
        model_sfco2, socat_sfco2, and mask data (that also contains sea mask)
    """
        
    da_list = conform_inputs_to_gcb_format(sfco2)
    da = da_list[0].astype(float)
    check_lon_alignment(da)
    check_pco2_values(da)
    results = socat_evaluation(da, method=method, cache_dir=cache_dir, region=region)
    return results.astype(float)


def _socat_evaluation_all(
    sfco2, method="annual", cache_dir=f"{BASE}/data_cache/"
):

    eval_func = _socat_evaluation_region
    drop = ["mask", "sfco2_socat", "sfco2_input"]
    socat_version = get_socat().version

    log_func(25, f"Comparing sfco2 against {socat_version} for region {'GLOBAL'}")
    ds = eval_func(
        sfco2, region="global", method=method, cache_dir=cache_dir
    )
    
    rmse = ds.drop(drop).expand_dims(region=["global"])

    for r in ["north", "tropics", "south"]:
        log_func(25, f"Comparing sfco2 against {socat_version} for region {r.upper()}")
        reg = eval_func(sfco2, region=r, method=method, cache_dir=cache_dir)
        reg = reg.drop(drop)
        reg = reg.expand_dims(region=[r])
        rmse = xr.concat([rmse, reg], "region")

    ds = xr.merge([ds[drop], rmse]).load()
    ds["mask"] = (ds.mask * get_regions(ds.mask)).astype(int)
    return ds


def _socat_evaluation_region(
    sfco2,
    region="global",
    method="annual",
    cache_dir=f"{BASE}/data_cache/",
):
    """
    Calculates the evaluation statistics for sfCO2 data (RMSE and bias).

    Parameters
    ----------
    sfco2 : xr.DataArray | np.ndarray
        the sfCO2 data that has shape (time=>384, lat=180, lon=360)
    region : int | str
        choose from one of three regions. global=0, north=1, tropics=2, south=3, all=4
        can be a string or an integer value
    method : str ('annual')
        choose from 'monthly' or 'annual' where these methods differ in the order
        the averaging is done
    cache_dir : str (f'BASE/data_cache/')
        the directory where the cache files are stored (e.g. SOCAT gridded)
        the BASE is defined as the root directory of GCBocean

    Returns
    -------
    xr.Dataset
        the evaluation statistics for sfCO2 (differs depending on the method)
        but will always contain RMSE and bias. Also contains the gridded
        model_sfco2, socat_sfco2, and mask data (that also contains sea mask)
    """

    log_func(10, "Fetching SOCAT data and region mask")
    socat_sfco2 = get_socat(cache_dir).sel(time=slice(None, f"{YEAR_END}-12"))
    mask = get_region_mask(region, cache_dir=cache_dir)
    log_func(20, f"Calculating statistics for region: {mask.region_name}")

    # make sure that the format of these inputs is correct
    # this is a bit slow, but it is robust
    yhat, y, m = conform_inputs_to_gcb_format(
        np.array(sfco2), socat_sfco2, mask
    )

    # final checking - will raise an error if the shapes are not the same
    xr.align(y, yhat, join="exact")
    xr.align(y, m, join="exact")

    # doing the masking before we calculate statistics
    ym = y.where(m)
    yhatm = yhat.where(m)
    
    if method == "monthly":
        stats = rmse_monthly(ym, yhatm)
        log_func(20, f"Using monthly point-wise RMSE method: {stats.rmse_avg.description}\n")
    elif method == "annual":
        stats = rmse_annual(ym, yhatm)
        log_func(20, f"Using annual time-series-based RMSE method {stats.rmse_avg.description}\n")
    else:
        raise ValueError(
            f"Method {method} not recognised. Choose 'monthly' or 'annual'"
        )

    ds = stats
    ds["sfco2_socat"] = y
    ds["sfco2_input"] = yhat
    ds["mask"] = m

    return ds


@wraps(_socat_evaluation_region)
def socat_evaluation(*args, **kwargs):
    """A wrapper for global and regional

    Returns
    -------
    xr.Dataset
        a dataset with the comparissons
    """
    region = kwargs.get("region", "all")
    if (str(region) == "all") or (str(region) == "4"):
        kwargs.pop("region", None)
        return _socat_evaluation_all(*args, **kwargs)
    else:
        return _socat_evaluation_region(*args, **kwargs)


def check_lon_alignment(pco2):
    """Sanity check for longitude alignment"""
    
    sahara = pco2.sel(lat=20.5, lon=20.5)
    pacific = pco2.sel(lat=0.5, lon=240.5)
    
    if not sahara.isnull().all():
        raise ValueError("There are pCO2 values over the Sahara; check 'lon' alignment. ")
    
    if pacific.isnull().all():
        raise ValueError("There are no values in the Equatorial Pacific. Check 'lon' alignment. ")


def check_pco2_values(sfco2):
    """Sanity check for sfco2 values (expected range = [0, 1000])"""
    
    vmin = np.nanmin(sfco2.values)
    vmax = np.nanmax(sfco2.values)
    avg = np.nanmean(sfco2.values)
    std = np.nanstd(sfco2.values)
    
    upper = round(min([avg + std * 3, vmax]), 2)
    lower = round(max([avg - std * 3, vmin]), 2)
    
    if (lower < 0) or (upper > 1000):
        raise ValueError(
            f"The input pCO2 contains values outside of the expected range "
            f"X = [{lower}, {upper}] E = [0, 1000]. "
            f"Please check that the input variable is pCO2 in uatm.")
    else:
        log_func(20, f"Input sfco2 is within expected range [0, 1000] - [{lower}, {upper}]. \n")


def rmse_annual(y, yhat):
    """
    Calculate RMSE by comparing regionally integrated time series of spco2
    GCB versions:
        - subsample data points for where there are obs
        - average over lon and lat
        - calculate annual mean
        - take out any point where model or data are NaNs (no need)
        - detrend both time-series (model, socat)
        - calculate rmse (using square root of the averaged square of residuals)

    Parameters
    ----------
    y : xr.DataArray
        the target data (e.g. SOCAT sfCO2)
    yhat : xr.DataArray
        the model data (e.g. IPSL sfCO2)

    Returns
    -------
    dict: {'diff_an': xr.DataArray 'rmse': xr.DataArray, 'bias': xr.DataArray}
        diff_an: the difference between the model and the target data (annualised)
        rmse: time averaged rmse (but annual average taken first)
        bias: yhat - y (same as rmse but only the mean)
    """
    # bug fix: mask SOCAT and estimates
    # need to mask the data before we do the averaging spatially, 
    # otherwise, there might be some regions in the SOCAT data that 
    # are not covered by the model and this will skew the results
    # get the mask of nans for both the model and the data
    nanmask = y.notnull() & yhat.notnull()
    # apply the mask to both the model and the data
    yhat_subs = yhat.where(nanmask)
    y = y.where(nanmask)

    y_avg = y.mean(["lat", "lon"])
    yhat_subs = yhat.where(y.notnull())
    yhat_avg = yhat_subs.mean(["lat", "lon"])
    count = y.count(["lat", "lon"])

    y_an = y_avg.groupby("time.year").mean("time")
    yhat_an = yhat_avg.groupby("time.year").mean("time")
    count_an = count.groupby("time.year").sum("time")

    y_an_detrend = calc_detrended(y_an, dim="year")
    yhat_an_detrend = calc_detrended(yhat_an, dim="year")

    diff = yhat_an_detrend - y_an_detrend

    stats = dict(
        residuals=diff,
        counts_socat=count_an,
        bias_avg=diff.mean("year"),
        rmse_avg=(diff**2).mean("year") ** 0.5,
    )

    stats = dict_to_dataset(stats)
    stats.residuals.attrs = dict(
        description="difference between model and socat detrended time series where SOCAT has data"  # noqa
    )
    stats.counts_socat.attrs = dict(description="number of socat points in each year")
    stats.bias_avg.attrs = dict(
        description="bias between model and socat detrended time series"
    )
    stats.rmse_avg.attrs = dict(
        description="fco2[time, lat, lon] --> mean(lat, lon) -> annual_mean -> detrend -> (fco2 - socat) -> square(resid) -> mean -> sqrt"  # noqa
    )

    return stats


def rmse_monthly(y, yhat):
    """
    Calculate RMSE based on monthly data
        - difference between the model and target data = residuals
        - square the resudials
        - average over lat and lon
        - resample to annual mean
        - square root of the annual mean
        - average over time (years)

    Parameters
    ----------
    y : xr.DataArray
        the target data (e.g. SOCAT sfCO2)
    yhat : xr.DataArray
        the model data (e.g. IPSL sfCO2)

    Returns
    -------
    dict: {'rmse_an': xr.DataArray 'rmse': xr.DataArray, 'bias': xr.DataArray}
        rmse_an: Annual RMSE values
        bias_an: Annual bias values
        rmse: average of rmse_an
        bias: average of bias_an

    """
    # masking isn't required for rmse_monthly since a point-wise comparison is done
    # meaning that missing points are ignored 

    # doing the maths
    diff = yhat - y
    count = diff.count(["lat", "lon"])
    rmse = (diff**2).mean(["lat", "lon"])
    bias = diff.mean(["lat", "lon"])

    # stats resampled to one year periods
    rmse_an = rmse.groupby("time.year").mean("time") ** 0.5
    bias_an = bias.groupby("time.year").mean("time")
    count_an = count.groupby("time.year").sum("time")

    stats = dict_to_dataset(
        dict(
            rmse=rmse_an,
            bias=bias_an,
            counts_socat=count_an,
            rmse_avg=rmse_an.mean("year"),
            bias_avg=bias_an.mean("year"),
        )
    )

    stats.rmse.attrs = dict(
        description="(fco2 - socat) -> resid[time,lat,lon] -> square(resid) -> mean(lat, lon) -> annual_mean -> sqrt"  # noqa
    )
    stats.bias.attrs = dict(
        description="(fco2 - socat) -> resid[time,lat,lon] -> mean(lat, lon) -> annual_mean"  # noqa
    )
    stats.counts_socat.attrs = dict(description="number of socat grid points in each year")
    stats.rmse_avg.attrs = dict(
        description="(fco2 - socat) -> resid[time,lat,lon] -> square(resid) -> mean(lat, lon) -> annual_mean -> sqrt -> mean"  # noqa
    )
    stats.bias_avg.attrs = dict(
        description="(fco2 - socat) -> resid[time,lat,lon] -> mean(lat, lon) -> annual_mean -> mean"  # noqa
    )

    return stats


def dict_to_dataset(dictionary):
    ds = xr.Dataset()
    for key, val in dictionary.items():
        if isinstance(val, xr.DataArray):
            ds[key] = val
        elif isinstance(val, pd.Series):
            ds[key] = val.to_xarray()
        else:
            raise TypeError(f"{key} is not a DataArray or Series (contact Luke)")

    return ds


@lru_cache(maxsize=1)
def get_socat(cache_dir=f"{BASE}/data_cache/"):
    """
    Returns the SOCAT data.
    """

    socat = _get_socat(cache_dir).load()
    return socat


@lru_cache(maxsize=1)
def get_seamask(cache_dir=f"{BASE}/data_cache/"):
    """
    Returns the SeaMask data.
    """

    seamask = _get_woa13_land_sea_mask(cache_dir).astype(bool).load()
    return seamask


def get_region_mask(region: int, cache_dir=f"{BASE}/data_cache/"):
    """
    Returns a mask for the region with a seamask applied

    Parameters
    ----------
    region : int | str
        the region to get the mask for can be either an integer or a string
        with the following values: 0=global, 1=north, 2=tropics, 3=south
    cache_dir : str
        the directory to download the woa13 sea mask to

    Returns
    -------
    xr.DataArray
        the mask for the selected region as a DataArray (lat, lon)
    """

    seamask = get_seamask(cache_dir)
    regions_mask = get_regions(seamask)

    names = np.array(["GLOBAL"] + [s.upper() for s in regions_mask.region_names])
    if isinstance(region, int):
        reg_int = region
    elif isinstance(region, str):
        region_upper = region.upper()
        reg_int = np.where(names == region_upper)[0]
        if len(reg_int) != 1:
            raise ValueError(f"Region `{region}` not found")
        else:
            reg_int = int(reg_int[0])
    else:
        raise TypeError(f"region must be int or str, not {type(region)}")

    assert 0 <= reg_int <= 3, "region must be between 0 and 3"
    reg_str = names[reg_int]
    if reg_str == "GLOBAL":
        mask = seamask
    else:
        mask = (regions_mask == reg_int) & seamask
    mask = mask.assign_attrs(region_name=reg_str)

    # this cannot run at the moment, but could be useful later
    if (region == 4) or (region == "ALL"):
        mask = regions_mask.where(seamask)
        mask = mask.assign_attrs(region_name=["GLOBAL", "NORTH", "TROPICS", "SOUTH"])

    return mask


def conform_inputs_to_gcb_format(*args) -> list:
    """
    Takes any number of array-like inputs and makes them GCB format

    Parameters
    ----------
    args : list of array-like
        arrays can be xr.DataArray or np.ndarray types.
        Assume that they are either 3D (time, lat, lon) or 2D (lat, lon)
        If any of the inputs is an xr.DataArray, all other non-xr.DataArray
        inputs will receive the same coordinates if the shapes are the same.

    Returns
    -------
    list:
        xr.DataArray with coordinates (time, lat, lon)
    """

    # check shapes to make sure they are the same
    dim_matching = dict(
        lat=all([args[0].shape[-2] == a.shape[-2] for a in args]),
        lon=all([args[0].shape[-1] == a.shape[-1] for a in args]),
    )
    for key in dim_matching:
        if not dim_matching[key]:
            shapes = str([a.shape for a in args])[1:-1]
            raise ValueError(f"{key} dimensions do not match: {shapes}")
    log_func(10, "Lat and Lon dimensions are the right size")

    args = np.array(args, dtype="object")
    is_xda = [isinstance(x, xr.DataArray) for x in args]
    is_arr = [isinstance(x, np.ndarray) for x in args]

    if (not any(is_xda)) and (not any(is_arr)):
        # if inputs are not right, we raise an error
        raise TypeError("Must pass xarray.DataArray or numpy.ndarray")

    # the goal here is to make all inputs data arrays
    if all(is_xda):
        log_func(20, "All inputs are xarray.DataArray, no changes made")
        new_args = args
    else:
        new_args = np.ndarray([len(args)], dtype="object")
        for i, da in enumerate(args):
            new_args[i] = convert_array_to_xarray_based_on_shape(da)
        log_func(20, f"Converted monthly input to xr.DataArray based on shape (assumed end year = {YEAR_END})")

    # choosing overlapping time steps only
    # find the overlapping time steps (max t0, min t1)
    tmin = np.datetime64("1900-01-01")
    tmax = pd.Timestamp.now().to_datetime64()
    for da in new_args:
        if "time" in da.coords:
            t0, t1 = da.time.values[[0, -1]]
            tmin = max([t0, tmin])
            tmax = min([t1, tmax])
    tmin, tmax = [pd.Timestamp(t) for t in [tmin, tmax]]
    log_func(20, f"Common time period: {tmin:%Y-%m} - {tmax:%Y-%m}")
    if tmax.year < YEAR_END:
        raise ValueError(
            f"Lowest year in data is {tmax.year}, but must be {YEAR_END} for GCB 2022"
        )

    for i, da in enumerate(new_args):
        if "time" in da.coords:
            new_args[i] = da.sel(time=slice(tmin, tmax))

    return new_args


def convert_array_to_xarray_based_on_shape(array):
    """Converts the input array to [time, lat, lon] xr.DataArray
    
    Assumes that time = [???? : YYYY-12-15], lat = [-89.5 : 89.5], lon = [0.5, 359.5]

    Parameters
    ----------
    array : np.ndarray
        The input array that should have shape <?>, 180, 360

    Returns
    -------
    xr.DataArray
        A data array with the associated coordinates
    """

    assert array.shape[-2] == 180, "Latitude dimension (1) must be 180"
    assert array.shape[-1] == 360, "Longitude dimension (2) must be 360"

    log_func(10, "Assuming lat = [-89.5 : 1 : 89.5] and lon = [0.5 : 1 : 359.5]")
    lat = np.arange(-89.5, 90)
    lon = np.arange(0.5, 360)

    if len(array.shape) == 3:
        t0 = f"{YEAR_END}-12-01"  # starting time
        td = pd.Timedelta("14D")  # time delta
        tn = array.shape[0]  # time number of points
        time = pd.date_range(t0, periods=tn, freq="-1MS")[::-1] + td
        coords = dict(time=time, lat=lat, lon=lon)
        dims = ["time", "lat", "lon"]
        log_func(
            10,
            "Input array is 3D. dim0 = time "
            f"[{time[0]:%Y-%m} : 1M : {time[-1]:%Y-%m}; n = {tn}] ",
        )
    elif len(array.shape) == 2:
        coords = dict(lat=lat, lon=lon)
        dims = ["lat", "lon"]
    else:
        raise ValueError("Array must be 2D or 3D")
    da = xr.DataArray(array, coords=coords, dims=dims)

    return da


def silence_warnings(func):
    """
    Silences warnings for the duration of the function.
    """
    import warnings
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True):
            out = func(*args, **kwargs)
            return out

    return wrapper


@silence_warnings
def calc_trend(da: xr.DataArray, dim="time"):
    coeffs = da.polyfit(dim, deg=1)
    trend = xr.polyval(da[dim], coeffs)
    return trend


@silence_warnings
def calc_detrended(da: xr.DataArray, dim="time"):
    trend = calc_trend(da, dim)
    detrended = da - trend
    return detrended.polyfit_coefficients


@silence_warnings
def calc_iav(da: xr.DataArray, dim_avg=["lat", "lon"]):
    """Calculates the IAV"""
    hist = ""
    name = da.name
    if dim_avg is not None:
        (da,) = conform_inputs_to_gcb_format(da)
        da = da.mean(dim_avg)
        hist += f"data ({name}) was averaged across {dim_avg} -> "
    da_annual = da.resample(time="1AS").mean(dim="time")
    detrended = calc_detrended(da_annual, "time")
    hist += (
        f"`{name}` resampled from monthly to annual "
        "-> detreneded along time dimension "
        "-> standard deviation along time dimension taken"
    )
    iav = detrended.std("time").assign_attrs(history=hist)
    return iav


def get_regions(da):
    """
    Get a mask for the regions
    """
    if "time" in da.coords:
        da = da.isel(time=0, drop=True)

    mask = xr.zeros_like(da) + 2
    mask += (da.lat > 30).astype(int) * -1
    mask += (da.lat < -30).astype(int) * 1

    mask.attrs["region_names"] = ["north", "tropics", "south"]

    return mask


def _get_socat(save_dir="."):
    from pandas import Timedelta
    import pooch
    import re

    filename = SOCAT_URL.split("/")[-1]
    fname = pooch.retrieve(
        SOCAT_URL,
        None,
        path=save_dir,
        fname=filename,
        progressbar=True,
        processor=pooch.Unzip(),
    )[0]

    version = re.findall("SOCATv[0-9]{4}", fname)
    if len(version) > 0:
        version = version[0]
    else:
        version = "SOCATv????"
    ds = xr.open_mfdataset(fname)
    time = ds.tmnth.values.astype("datetime64[M]") + Timedelta(days=14)
    da = (
        ds.fco2_ave_unwtd.rename(tmnth="time", xlon="lon", ylat="lat")
        .assign_coords(time=time)
        .pipe(transform_lon)
        .assign_attrs(
            units="uatm",
            long_name="Sea surface fugacity of CO2",
            version=version,
        )
    )

    return da


def transform_lon(da, lon_name="lon"):
    """
    Transform longitude to the range [0, 360]
    """
    return da.assign_coords(**{lon_name: da[lon_name] % 360}).sortby(lon_name)


def _get_woa13_land_sea_mask(save_dir="."):
    import pooch
    from pandas import read_csv

    url = "https://www.ncei.noaa.gov/data/oceans/woa/WOA13/MASKS/landsea_01.msk"
    fname = pooch.retrieve(url, None, path=save_dir, fname="WOA13_landsea_01.msk")

    df = read_csv(fname, sep=",", header=1, index_col=[0, 1])
    depth_level = (
        df.to_xarray()
        .Bottom_Standard_Level.rename(Latitude="lat", Longitude="lon")
        .pipe(transform_lon)
    )

    mask = depth_level > 1
    mask = mask.astype(float)

    return mask


def download(url, dest_path='.', fname=None, progressbar=True):
    from pooch import retrieve
    
    if fname is None:
        fname = url.split('/')[-1].split('?')[0]
    dest = retrieve(url, None, fname, dest_path, progressbar=progressbar)

    return dest
