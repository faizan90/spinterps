'''
Created on May 27, 2019

@author: Faizan-Uni
'''

from pathlib import Path

import gdal
import h5py
import numpy as np

from ..misc import print_sl, print_el


class ExtractGTiffCoords:

    _raster_type_lab = 'gtiff'

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._x_crds = None
        self._y_crds = None

        self._set_in_flag = False
        return

    def set_input(self, path_to_gtiff):

        assert isinstance(path_to_gtiff, (str, Path))

        path_to_gtiff = Path(path_to_gtiff).absolute()

        assert path_to_gtiff.exists()

        self._in_path = path_to_gtiff

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the GTiff:')
            print(f'Path: {self._in_path}')

            print_el()

        self._set_in_flag = True
        self._set_crds_extrt_flag = True
        return

    def extract_coordinates(self):

        assert self._set_in_flag

        in_hdl = gdal.Open(str(self._in_path))

        assert in_hdl is not None

        assert in_hdl.GetDriver().ShortName == 'GTiff'

        n_rows = in_hdl.RasterYSize
        n_cols = in_hdl.RasterXSize

        assert n_rows
        assert n_cols

        geotransform = in_hdl.GetGeoTransform()

        in_hdl = None

        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        assert pix_width > 0
        assert pix_height > 0

        x_max = x_min + (n_cols * pix_width)
        y_min = y_max - (n_rows * pix_height)

        assert x_min < x_max
        assert y_min < y_max

        assert np.all(np.isfinite([x_min, x_max, y_min, y_max]))

        self._x_crds = np.linspace(x_min, x_max, n_cols + 1)
        self._y_crds = np.linspace(y_max, y_min, n_rows + 1)

        assert self._x_crds.shape[0] == (n_cols + 1)
        assert self._y_crds.shape[0] == (n_rows + 1)

        assert np.all(np.isfinite(self._x_crds))
        assert np.all(np.isfinite(self._y_crds))

        assert (
            np.all(np.ediff1d(self._x_crds) > 0) or
            np.all(np.ediff1d(self._x_crds[::-1]) > 0))

        assert (
            np.all(np.ediff1d(self._y_crds) > 0) or
            np.all(np.ediff1d(self._y_crds[::-1]) > 0))

        if self._vb:
            print_sl()

            print(f'INFO: GTiff coordinates\' properties:')
            print(f'Shape of X coordinates: {self._x_crds.shape}')
            print(f'Shape of Y coordinates: {self._y_crds.shape}')

            print_el()

        self._set_crds_extrt_flag = True
        return

    def get_x_coordinates(self):

        assert self._set_crds_extrt_flag

        return self._x_crds

    def get_y_coordinates(self):

        assert self._set_crds_extrt_flag

        return self._y_crds


class ExtractGTiffValues:

    def __init__(self, verbose=True):

        self._vb = verbose

        self._in_path = None
        self._extrtd_data = None

        self._out_path = None
        self._out_fmt = None

        self._set_in_flag = False
        self._set_out_flag = False
        self._set_data_extrt_flag = False
        return

    def set_input(self, path_to_gtiff):

        assert isinstance(path_to_gtiff, (str, Path))

        path_to_gtiff = Path(path_to_gtiff).absolute()

        assert path_to_gtiff.exists()

        self._in_path = path_to_gtiff

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the GTiff:')
            print(f'Path: {self._in_path}')

            print_el()

        self._set_in_flag = True
        return

    def set_output(self, path_to_output=None):

        if path_to_output is None:
            self._out_fmt = 'raw'

        else:
            assert isinstance(path_to_output, (str, Path))

            path_to_output = Path(path_to_output).absolute()

            assert path_to_output.parents[0].exists()

            fmt = path_to_output.suffix

            if fmt in ('.h5', '.hdf5'):
                self._out_fmt = 'h5'

            else:
                raise NotImplementedError

        self._out_path = path_to_output

        if self._vb:
            print_sl()

            print(f'INFO: Set the following parameters for the output:')
            print(f'Path: {self._out_path}')
            print(f'Format: {self._out_fmt}')

            print_el()

        self._set_out_flag = True
        return

    def _verf_idxs(self, indicies, save_add_vars_flag):

        add_var_labels_main = set()

#         for label, crds_idxs in indicies.items():
        for crds_idxs in indicies.values():
            assert 'cols' in crds_idxs
            cols_idxs = crds_idxs['cols']
            assert cols_idxs.ndim == 1
            assert cols_idxs.size > 0
            assert np.issubdtype(cols_idxs.dtype, np.integer)

            assert 'rows' in crds_idxs
            rows_idxs = crds_idxs['rows']
            assert rows_idxs.ndim == 1
            assert rows_idxs.size > 0
            assert np.issubdtype(rows_idxs.dtype, np.integer)

            if not save_add_vars_flag:
                continue

            if not add_var_labels_main:
                add_var_labels_main = set(
                    crds_idxs.keys()) - set(('rows', 'cols'))

            else:
                assert not (
                    add_var_labels_main -
                    set(crds_idxs.keys()) -
                    set(('rows', 'cols')))

        return add_var_labels_main

    def extract_data_for_indices_and_save(
            self, indicies, save_add_vars_flag=True):

        assert self._set_in_flag
        assert self._set_out_flag

        assert isinstance(indicies, dict)
        assert indicies

        assert isinstance(save_add_vars_flag, bool)

        add_var_labels = self._verf_idxs(indicies, save_add_vars_flag)

        in_hdl = gdal.Open(str(self._in_path))

        assert in_hdl is not None

        assert in_hdl.GetDriver().ShortName == 'GTiff'

        n_bnds = in_hdl.RasterCount

        assert n_bnds > 0

        gtiff_bnds = {}
        for i in range(1, n_bnds + 1):
            bnd = in_hdl.GetRasterBand(i)

            ndv = bnd.GetNoDataValue()

            data = bnd.ReadAsArray().astype(float)

            assert data.ndim == 2
            assert data.size > 0

            if not np.isnan(ndv):
                data[np.isclose(data, ndv)] = np.nan

            assert np.any(np.isfinite(data))

            gtiff_bnds[f'B{i:02d}'] = data

        in_hdl = None

        assert gtiff_bnds

        if self._vb:
            print_sl()

            print(f'INFO: Read {n_bnds} from the input raster')

            print_el()

        path_stem = self._in_path.stem
        assert path_stem

        misc_grp_labs = add_var_labels | set(('rows', 'cols'))

        extrtd_data = {}

        if self._out_fmt == 'h5':
            out_hdl = h5py.File(str(self._out_path), mode='a', driver=None)

            if 'rows' not in out_hdl:
                for grp in misc_grp_labs:
                    assert grp not in out_hdl
                    out_hdl.create_group(grp)

            else:
                for grp in misc_grp_labs:
                    assert grp in out_hdl

            assert path_stem not in out_hdl

            out_var_grp = out_hdl.create_group(path_stem)

        elif self._out_fmt == 'raw':
            pass

        else:
            raise NotImplementedError

        for label, crds_idxs in indicies.items():

            cols_idxs = crds_idxs['cols']
            cols_idxs_min = cols_idxs.min()
            cols_idxs_max = cols_idxs.max()
            assert (cols_idxs_max > 0) & (cols_idxs_min > 0)

            rows_idxs = crds_idxs['rows']
            rows_idxs_min = rows_idxs.min()
            rows_idxs_max = rows_idxs.max()
            assert (rows_idxs_max > 0) & (rows_idxs_min > 0)

            assert cols_idxs.shape == rows_idxs.shape

            crds_set = set([(x, y) for x, y in zip(cols_idxs, rows_idxs)])

            assert len(crds_set) == cols_idxs.size

            bnds_data = {}
            for bnd, data in gtiff_bnds.items():
                assert rows_idxs_max < data.shape[0]
                assert cols_idxs_max < data.shape[1]

                bnd_data = data[rows_idxs, cols_idxs]

                bnds_data[bnd] = bnd_data

            assert bnds_data

            extrtd_data[label] = bnds_data

            if self._out_fmt == 'raw':
                pass

            elif self._out_fmt == 'h5':
                label_str = str(label)

                assert label_str not in out_var_grp

                for add_var_lab in add_var_labels:
                    grp_lnk = f'{add_var_lab}/{label_str}'

                    if label_str in out_hdl[add_var_lab]:
                        assert np.all(np.isclose(
                            out_hdl[grp_lnk][...],
                            crds_idxs[add_var_lab]))

                    else:
                        out_hdl[grp_lnk] = crds_idxs[add_var_lab]

                if label_str in out_hdl['rows']:
                    assert np.all(
                        out_hdl[f'rows/{label_str}'][...] == rows_idxs)

                    assert np.all(
                        out_hdl[f'cols/{label_str}'][...] == cols_idxs)

                else:
                    out_hdl[f'rows/{label_str}'] = rows_idxs
                    out_hdl[f'cols/{label_str}'] = cols_idxs

                out_var_grp[label_str] = np.vstack(list(bnds_data.values()))

                out_hdl.flush()

            else:
                raise NotImplementedError

        gtiff_bnds = None

        if self._out_fmt == 'h5':
            out_hdl.flush()
            out_hdl.close()
            out_hdl = None

        elif self._out_fmt == 'raw':
            assert extrtd_data
            self._extrtd_data = extrtd_data

        else:
            raise NotImplementedError

        self._set_data_extrt_flag = True
        return

    def get_extracted_data(self):

        assert self._set_data_extrt_flag

        return self._extrtd_data
