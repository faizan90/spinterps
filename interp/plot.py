'''
Created on Apr 28, 2022

@author: Faizan3800X-Uni
'''

import warnings

import numpy as np
import netCDF4 as nc

from descartes import PolygonPatch
import matplotlib.pyplot as plt; plt.ioff()


class SpInterpPlot:

    def __init__(self, spinterp_main_cls):

        read_labs = [
            '_vb',
            '_n_cpus',
            '_mp_flag',
            '_crds_df',
            '_nc_vlab',
            '_nc_vunits',
            '_plot_polys',
            '_interp_crds_orig_shape',
            '_interp_x_crds_plt_msh',
            '_interp_y_crds_plt_msh',
            '_index_type',
            '_nc_file_path',
            ]

        for read_lab in read_labs:
            setattr(self, read_lab, getattr(spinterp_main_cls, read_lab))

        return

    def plot_interp_steps(self, args):

        (data_df,
         beg_nc_idx,
         end_nc_idx,
         interp_args,
         vgs_ser) = args

        out_figs_dirs = {
            interp_arg[2]: interp_arg[1] for interp_arg in interp_args}

        interp_types = [interp_arg[0] for interp_arg in interp_args]
        interp_labels = [interp_arg[2] for interp_arg in interp_args]

        time_steps = data_df.index

        nc_hdl = nc.Dataset(str(self._nc_file_path), mode='r')

        if vgs_ser is not None:
            vgs_ser = vgs_ser.loc[data_df.index]

        for interp_type, interp_label in zip(interp_types, interp_labels):
            out_figs_dir = out_figs_dirs[interp_label]

            for i, nc_idx in enumerate(range(beg_nc_idx, end_nc_idx)):
                if interp_type in ('IDW', 'NNB'):
                    model = None

                elif interp_type in ('OK', 'SK', 'EDK'):
                    model = vgs_ser.loc[time_steps[i]]

                else:
                    raise NotImplementedError(interp_type)

                interp_fld = nc_hdl[interp_label][nc_idx].data

                self._plot_interp_step(
                    interp_fld,
                    self._crds_df.loc[:, 'X'].values,
                    self._crds_df.loc[:, 'Y'].values,
                    time_steps[i],
                    model,
                    interp_type,
                    out_figs_dir,
                    data_df.loc[time_steps[i]].values)

        nc_hdl.close()
        return

    def _plot_interp_step(
            self,
            interp_fld,
            curr_x_coords,
            curr_y_coords,
            interp_time,
            model,
            interp_type,
            out_figs_dir,
            data_vals):

        if self._index_type == 'date':
            time_str = interp_time.strftime('%Y_%m_%d_T_%H_%M')

        elif self._index_type == 'obj':
            time_str = interp_time

        else:
            raise NotImplementedError(
                f'Unknown index_type: {self._index_type}!')

        out_fig_name = f'{interp_type.lower()}_{time_str}.png'

        fig, ax = plt.subplots()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            data_min = np.nanmin(data_vals)
            data_max = np.nanmax(data_vals)
            data_mean = np.nanmean(data_vals)
            data_var = np.nanvar(data_vals)

            grd_min = np.nanmin(interp_fld)
            grd_max = np.nanmax(interp_fld)
            grd_mean = np.nanmean(interp_fld)
            grd_var = np.nanvar(interp_fld)

        pclr = ax.pcolormesh(
            self._interp_x_crds_plt_msh,
            self._interp_y_crds_plt_msh,
            interp_fld,
            vmin=grd_min,
            vmax=grd_max,
            shading='flat')

        cb = fig.colorbar(pclr)

        cb.set_label(self._nc_vlab + ' (' + self._nc_vunits + ')')

        if False:
            ax.scatter(
                curr_x_coords,
                curr_y_coords,
                label='obs. pts.',
                marker='+',
                c='r',
                alpha=0.7)

            ax.legend(framealpha=0.5, loc=1)

        if self._plot_polys is not None:
            for poly in self._plot_polys:
                ax.add_patch(PolygonPatch(poly, alpha=1, fc='None', ec='k'))

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        title_str = f'Step: {time_str}\n'

        if model is not None:
            title_str += f'(VG: {model})\n'

        title_str += (
            f'Data stats:: Mean: {data_mean:0.4f}, '
            f'Var.: {data_var:0.4f}, '
            f'Min.: {data_min:0.4f}, '
            f'Max.: {data_max:0.4f}\n')

        title_str += (
            f'Grid stats:: Mean: {grd_mean:0.4f}, '
            f'Var.: {grd_var:0.4f}, '
            f'Min.: {grd_min:0.4f}, '
            f'Max.: {grd_max:0.4f}\n')

        ax.set_title(title_str)

        plt.setp(ax.get_xmajorticklabels(), rotation=70)
        ax.set_aspect('equal', 'datalim')

        plt.savefig(str(out_figs_dir / out_fig_name), bbox_inches='tight')
        plt.close()
        return
