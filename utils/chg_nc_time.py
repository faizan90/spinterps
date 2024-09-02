# -*- coding: utf-8 -*-

'''
@author: Faizan-TU Munich

01.07.2024

11:50:38

Description:

Keywords:

'''

import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import netCDF4 as nc

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\dwd_meteo\gridded\extract_radolan\a_snip_hourly')
    os.chdir(main_dir)

    ncs_dir = main_dir

    new_units = 'hours since 2006-01-01 00:00:00'
    new_clndr = 'gregorian'

    time_labl = 'time'
    time_strs_labl = 'time_strs'  # None

    glob_patt = r'*.nc'

    time_dlta = pd.Timedelta(-50, unit='minutes')
    #==========================================================================

    assert ncs_dir.exists(), ncs_dir

    for ncf_pth in ncs_dir.glob(glob_patt):

        print(ncf_pth)

        with nc.Dataset(ncf_pth, 'r+') as ncf_hdl:

            time_axis = ncf_hdl[time_labl]

            time_nums = time_axis[:]

            time_objs = nc.num2date(
                time_nums,
                units=time_axis.units,
                calendar=time_axis.calendar,
                only_use_cftime_datetimes=False) + time_dlta

            new_time_nums = nc.date2num(
                time_objs,
                units=new_units,
                calendar=new_clndr)

            time_axis.units = new_units
            time_axis.calendar = new_clndr

            time_axis[:] = new_time_nums

            if time_strs_labl is not None:
                time_objs = pd.DatetimeIndex(time_objs)

                time_strs_axis = ncf_hdl[time_strs_labl]
                time_strs_axis[:] = (
                    time_objs.strftime('%Y%m%dT%H%M%S').values)

    return


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
