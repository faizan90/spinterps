# -*- coding: utf-8 -*-

'''
@author: Faizan-Uni-Stuttgart

Dec 22, 2022

10:59:46 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\cmip6\ec-earth3-cc\bw')
    os.chdir(main_dir)

    # ncs_dir = Path(r'1960-2015_precip_g_rate_1h_snipped')
    # out_nc_path = ('ppt_1960_2015.nc')
    # var_labs = ['precip_g_rate']

    ncs_dir = main_dir  # All nc_ext files are used.
    out_nc_path = ('pr_1950_2014_bw.nc')
    var_labs = ['pr']

    nc_ext = 'nc'
    time_lab = 'time'
    x_coords_lab = 'x_utm32n'
    y_coords_lab = 'y_utm32n'
    #==========================================================================

    path_to_ncs = list(ncs_dir.glob('*.%s' % nc_ext))

    ncs_beg_time = get_nc_time_str(path_to_ncs[+0], time_lab, +0)
    ncs_end_time = get_nc_time_str(path_to_ncs[-1], time_lab, -1)

    out_nc_init_flag = False

    x_dim_lab = 'dimx'
    y_dim_lab = 'dimy'
    t_dim_lab = 'dimt'

    with nc.Dataset(out_nc_path, 'w') as out_nc_hdl:

        out_nc_hdl.set_auto_mask(False)

        for path_to_nc in path_to_ncs:

            print(path_to_nc.stem)

            with nc.Dataset(path_to_nc) as nc_hdl:

                time_axis = nc_hdl[time_lab]

                if not out_nc_init_flag:

                    out_nc_hdl.createDimension(
                        x_dim_lab, nc_hdl[x_coords_lab].shape[1])

                    out_nc_hdl.createDimension(
                        y_dim_lab, nc_hdl[y_coords_lab].shape[0])

                    out_nc_hdl.createDimension(t_dim_lab, None)

                    x_coords_nc = out_nc_hdl.createVariable(
                        x_coords_lab,
                        'd',
                        dimensions=(y_dim_lab, x_dim_lab))

                    y_coords_nc = out_nc_hdl.createVariable(
                        y_coords_lab,
                        'd',
                        dimensions=(y_dim_lab, x_dim_lab))

                    time_nc = out_nc_hdl.createVariable(
                        time_lab, 'i8', dimensions=t_dim_lab)

                    data_ncs_dict = {}
                    for var_lab in var_labs:
                        data_nc = out_nc_hdl.createVariable(
                            var_lab,
                            'd',
                            dimensions=(t_dim_lab, y_dim_lab, x_dim_lab))

                        data_ncs_dict[var_lab] = data_nc

                    data_nc = None

                    x_coords_nc[:] = nc_hdl[x_coords_lab][:]
                    y_coords_nc[:] = nc_hdl[y_coords_lab][:]

                    out_nc_time_units = time_axis.units
                    out_nc_time_calendar = time_axis.calendar

                    time_nc.units = out_nc_time_units
                    time_nc.calendar = out_nc_time_calendar

                    if 'hours' in out_nc_time_units:
                        time_freq = 'H'

                    elif 'days' in out_nc_time_units:
                        time_freq = 'D'

                    else:
                        raise NotImplementedError(
                            f'Unknown time units in {out_nc_time_units}!')

                    nc_time_idx_pydatetime = pd.date_range(
                        ncs_beg_time, ncs_end_time, freq=time_freq
                        ).to_pydatetime()

                    out_nc_time_idx = nc.date2num(
                        nc_time_idx_pydatetime,
                        units=out_nc_time_units,
                        calendar=out_nc_time_calendar)

                    time_nc[:] = out_nc_time_idx

                    out_nc_init_flag = True

                else:
                    assert time_nc.units == out_nc_time_units
                    assert time_nc.calendar == out_nc_time_calendar

                    assert np.all(
                        np.isclose(x_coords_nc[:], nc_hdl[x_coords_lab][:]))

                    assert np.all(
                        np.isclose(y_coords_nc[:], nc_hdl[y_coords_lab][:]))

                time_vals = time_axis[:].data.astype(out_nc_time_idx.dtype)

                assert np.in1d(time_vals, out_nc_time_idx).all()

                nc_time_insert_idxs = (time_vals - out_nc_time_idx[0]).astype(
                    np.int64)

                for var_lab in data_ncs_dict:
                    data_ncs_dict[var_lab][nc_time_insert_idxs,:,:] = (
                        nc_hdl[var_lab][:])

    return


def get_nc_time_str(path_to_nc, time_lab, idx):

    with nc.Dataset(path_to_nc) as nc_hdl:

        time_axis = nc_hdl[time_lab]

        time_vals = time_axis[[idx]]

        time_str = num2date(
            time_vals, time_axis.units, time_axis.calendar)[0].strftime()

    return time_str


def add_month(date, months_to_add):

    """
    Finds the next month from date.

    :param cftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int months_to_add: The number of months to add to the date
    :returns: The final date
    :rtype: *cftime.datetime*
    """

    years_to_add = int((
        date.month +
        months_to_add -
        np.mod(date.month + months_to_add - 1, 12) - 1) / 12)

    new_month = int(np.mod(date.month + months_to_add - 1, 12)) + 1

    new_year = date.year + years_to_add

    date_next = datetime(
        year=new_year,
        month=new_month,
        day=date.day,
        hour=date.hour,
        minute=date.minute,
        second=date.second)
    return date_next


def add_year(date, years_to_add):

    """
    Finds the next year from date.

    :param cftime.datetime date: Accepts datetime or phony datetime
        from ``netCDF4.num2date``.
    :param int years_to_add: The number of years to add to the date
    :returns: The final date
    :rtype: *cftime.datetime*
    """

    new_year = date.year + years_to_add

    date_next = datetime(
        year=new_year,
        month=date.month,
        day=date.day,
        hour=date.hour,
        minute=date.minute,
        second=date.second)
    return date_next


def num2date(num_axis, units, calendar):

    """
    A wrapper from ``nc.num2date`` able to handle "years since" and
        "months since" units.

    If time units are not "years since" or "months since", calls
    usual ``cftime.num2date``.

    :param numpy.array num_axis: The numerical time axis following units
    :param str units: The proper time units
    :param str calendar: The NetCDF calendar attribute
    :returns: The corresponding date axis
    :rtype: *array*
    """

    res = None
    if not units.split(' ')[0] in ['years', 'months']:
        res = nc.num2date(
            num_axis,
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=True)

    else:
        units_as_days = 'days ' + ' '.join(units.split(' ')[1:])

        start_date = nc.num2date(0.0, units=units_as_days, calendar=calendar)

        num_axis_mod = np.atleast_1d(np.array(num_axis))

        if units.split(' ')[0] == 'years':
            max_years = np.floor(np.max(num_axis_mod)) + 1
            min_years = np.ceil(np.min(num_axis_mod)) - 1

            years_axis = np.array([
                add_year(start_date, years_to_add)
                for years_to_add in np.arange(min_years, max_years + 2)])

            # cftime.utime is no longer supported.
            # cdftime = utime(units_as_days, calendar=calendar)
            cdftime = datetime.toordinal(units_as_days, calendar=calendar)

            years_axis_as_days = cdftime.date2num(years_axis)

            yind = np.vectorize(np.int)(np.floor(num_axis_mod))

            num_axis_mod_days = (
                years_axis_as_days[yind - int(min_years)] +
                (num_axis_mod - yind) *
                np.diff(years_axis_as_days)[yind - int(min_years)])

            res = nc.num2date(
                num_axis_mod_days, units=units_as_days, calendar=calendar)

        elif units.split(' ')[0] == 'months':
            max_months = np.floor(np.max(num_axis_mod)) + 1
            min_months = np.ceil(np.min(num_axis_mod)) - 1

            months_axis = np.array([
                add_month(start_date, months_to_add)
                for months_to_add in np.arange(min_months, max_months + 12)])

            # cftime.utime is no longer supported.
            # cdftime = utime(units_as_days, calendar=calendar)
            cdftime = datetime.toordinal(units_as_days, calendar=calendar)

            months_axis_as_days = cdftime.date2num(months_axis)

            mind = np.vectorize(np.int)(np.floor(num_axis_mod))

            num_axis_mod_days = (
                months_axis_as_days[mind - int(min_months)] +
                (num_axis_mod - mind) *
                np.diff(months_axis_as_days)[mind - int(min_months)])

            res = nc.num2date(
                num_axis_mod_days, units=units_as_days, calendar=calendar)

        else:
            raise ValueError(units.split(' ')[0])

    assert res is not None
    return res


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
