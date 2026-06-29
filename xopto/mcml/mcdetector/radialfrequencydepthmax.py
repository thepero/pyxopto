# -*- coding: utf-8 -*-
################################ Begin license #################################
# Copyright (C) Laboratory of Imaging technologies,
#               Faculty of Electrical Engineering,
#               University of Ljubljana.
#
# This file is part of PyXOpto.
#
# PyXOpto is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyXOpto is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyXOpto. If not, see <https://www.gnu.org/licenses/>.
################################# End license ##################################

from typing import Tuple

import numpy as np

from xopto.mcml.mcdetector.base import Detector
from xopto.mcml import cltypes, mcobject, mcoptions
from xopto.mcml.mcutil import axis


class RadialFrequencyDepthMax(Detector):
    @staticmethod
    def cl_type(mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClRadialFrequencyDepthMax(cltypes.Structure):
            '''
            Structure that represents a detector in the Monte Carlo
            simulator core.

            Fields
            ------
            direction: mc_point3f_t
                Reference direction/orientation of the detector.
            position: mc_point2f_t
                Position of the center/origin of the detector.
            f_min: mc_fp_t
                The first spatial frequency point (rad/m).
            df: mc_fp_t
                Separation of the spatial frequency points (rad/m).
            depthmax_min: mc_fp_t
                The leftmost edge of the first maximum depth accumulator (m).
            inv_ddepthmax: mc_fp_t
                Inverse of the width of the maximum depth accumulators (1/m).
            cos_min: mc_fp_t
                Cosine of the maximum acceptance angle (relative to the
                direction of the detector).
            n_f: mc_size_t
                The number of spatial frequency points.
            n_depthmax: mc_size_t
                The number of maximum depth accumulators.
            offset: mc_size_t
                The offset of the first accumulator in the Monte Carlo
                detector buffer.

            Note
            ----
            The buffer holds n_depthmax * 2*n_f accumulators.
            For depth index d and frequency index f:
              - positive J0 weight -> offset + d*(2*n_f) + 2*f
              - negative J0 weight -> offset + d*(2*n_f) + 2*f + 1
            Net SFD reflectance for each (d, f) bin: positive - negative.
            '''
            _fields_ = [
                ('direction', T.mc_point3f_t),
                ('position', T.mc_point2f_t),
                ('f_min', T.mc_fp_t),
                ('df', T.mc_fp_t),
                ('depthmax_min', T.mc_fp_t),
                ('inv_ddepthmax', T.mc_fp_t),
                ('cos_min', T.mc_fp_t),
                ('n_f', T.mc_size_t),
                ('n_depthmax', T.mc_size_t),
                ('offset', T.mc_size_t),
            ]
        return ClRadialFrequencyDepthMax

    def cl_declaration(self, mc: mcobject.McObject) -> str:
        '''
        Structure that defines the detector in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'struct MC_STRUCT_ATTRIBUTES Mc{}Detector{{'.format(Loc),
            '   mc_point3f_t direction;',
            '   mc_point2f_t position;',
            '   mc_fp_t f_min;',
            '   mc_fp_t df;',
            '   mc_fp_t depthmax_min;',
            '   mc_fp_t inv_ddepthmax;',
            '   mc_fp_t cos_min;',
            '   mc_size_t n_f;',
            '   mc_size_t n_depthmax;',
            '   mc_size_t offset;',
            '};'
        ))

    def cl_implementation(self, mc: mcobject.McObject) -> str:
        '''
        Implementation of the detector accumulator in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'void dbg_print_{}_detector(__mc_detector_mem const Mc{}Detector *detector){{'.format(loc, Loc),
            'dbg_print("Mc{}Detector - RadialFrequencyDepthMax detector:");'.format(Loc),
            'dbg_print_point3f(INDENT "direction:", &detector->direction);',
            'dbg_print_point2f(INDENT "position:", &detector->position);',
            'dbg_print_float(INDENT "f_min (1/mm):", detector->f_min*1e-3f);',
            'dbg_print_float(INDENT "df (1/mm):", detector->df*1e-3f);',
            'dbg_print_float(INDENT "depthmax_min (um):", detector->depthmax_min*1e6f);',
            'dbg_print_float(INDENT "inv_ddepthmax (1/um):", detector->inv_ddepthmax*1e-6f);',
            'dbg_print_float(INDENT "cos_min:", detector->cos_min);',
            'dbg_print_size_t(INDENT "n_f:", detector->n_f);',
            'dbg_print_size_t(INDENT "n_depthmax:", detector->n_depthmax);',
            'dbg_print_size_t(INDENT "offset:", detector->offset);',
            '};',
            '',
            'inline void mcsim_{}_detector_deposit('.format(loc),
            '    McSim *mcsim,',
            '    mc_point3f_t const *pos, mc_point3f_t const *dir,',
            '    mc_fp_t weight){',
            '',
            '    __global mc_accu_t *address;',
            '    mc_fp_t weight_sfd, frequency;',
            '    uint32_t ui32w;',
            '',
            '    dbg_print_status(mcsim, "{} RadialFrequencyDepthMax detector hit");'.format(Loc),
            '',
            '    __mc_detector_mem const struct Mc{}Detector *detector ='.format(Loc),
            '    mcsim_{}_detector(mcsim);'.format(loc),
            '',
            '    mc_fp_t dx = pos->x - detector->position.x;',
            '    mc_fp_t dy = pos->y - detector->position.y;',
            '    mc_fp_t r = mc_sqrt(dx*dx + dy*dy);',
            '',
            '    /* Compute depth_max bin index */',
            '    mc_fp_t depth = mcsim_depth_max(mcsim);',
            '    mc_int_t depth_index = mc_int((depth - detector->depthmax_min)*detector->inv_ddepthmax);',
            '    depth_index = mc_clip(depth_index, 0, (mc_int_t)(detector->n_depthmax - 1));',
            '',
            '    mc_point3f_t detector_direction = detector->direction;',
            '    mc_int_t directional_factor = (detector->cos_min <= mc_fabs(mc_dot_point3f(dir, &detector_direction))) ? 1 : 0;',
            '',
            '    /* Loop over spatial frequencies */',
            '    for (mc_int_t f_index = 0; f_index < (mc_int_t)detector->n_f; f_index++) {',
            '        frequency = detector->f_min + f_index*detector->df;',
            '        weight_sfd = weight * bessel_J0(FP_2PI * frequency * r);',
            '',
            '        /* Each (depth, frequency) uses two accumulators: [2*f] positive, [2*f+1] negative */',
            '        mc_size_t base = detector->offset + depth_index*(2*detector->n_f);',
            '        if (weight_sfd >= FP_0) {',
            '            address = mcsim_accumulator_buffer_ex(mcsim, base + 2*f_index);',
            '            ui32w = weight_to_int(weight_sfd) * directional_factor;',
            '        } else {',
            '            address = mcsim_accumulator_buffer_ex(mcsim, base + 2*f_index + 1);',
            '            ui32w = weight_to_int(-weight_sfd) * directional_factor;',
            '        }',
            '        if (ui32w > 0) {',
            '            dbg_print_uint("{} RadialFrequencyDepthMax detector depositing int:", ui32w);'.format(Loc),
            '            accumulator_deposit(address, ui32w);',
            '        }',
            '    }',
            '};'
        ))

    def cl_options(self, mc, target=None) -> mcoptions.RawOptions:
        '''
        OpenCL kernel options defined by this object.
        '''
        return [('MC_TRACK_DEPTH_MAX', True)]

    def __init__(self, faxis: axis.EdgeAxis,
                 depthmaxaxis: axis.Axis = None,
                 position: Tuple[float, float] = (0.0, 0.0),
                 cosmin: float = 0.0,
                 direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        '''
        Spatial-frequency / maximum-depth accumulator.

        Accumulates the SFD reflectance as a function of the maximum photon
        packet penetration depth. The raw buffer has shape
        ``(n_depthmax, 2*n_f)``: axis 0 is the depth_max bin, axis 1 encodes
        positive (even) and negative (odd) J0 weight pairs for each frequency.

        Parameters
        ----------
        faxis: axis.EdgeAxis
            Object that defines the spatial frequency points (rad/m).
        depthmaxaxis: axis.Axis
            Object that defines the maximum depth bins (m).
            Defaults to a single bin spanning [0, 1] m if not provided.
        position: (float, float)
            Position of the center of the detector (x, y) in meters.
        cosmin: float
            Cosine of the maximum acceptance angle relative to ``direction``.
        direction: (float, float, float)
            Reference direction / orientation of the detector.

        Note
        ----
        The first dimension of the raw accumulator is the depth_max axis,
        the second dimension is 2 * n_f (positive/negative J0 pairs).
        Net SFD reflectance: ``(raw[:, 0::2] - raw[:, 1::2]) / nphotons``.
        '''
        if isinstance(faxis, RadialFrequencyDepthMax):
            rfdm = faxis
            position = rfdm.position
            faxis = type(rfdm.faxis)(rfdm.faxis)
            depthmaxaxis = type(rfdm.depthmaxaxis)(rfdm.depthmaxaxis)
            cosmin = rfdm.cosmin
            direction = rfdm.direction
            raw_data = np.copy(rfdm.raw)
            nphotons = rfdm.nphotons
        else:
            if depthmaxaxis is None:
                depthmaxaxis = axis.Axis(0.0, 1.0, 1)
            raw_data = np.zeros((depthmaxaxis.n, 2 * faxis.n))
            nphotons = 0

        super().__init__(raw_data, nphotons)

        self._position = np.zeros((2,))
        self._cosmin = 0.0
        self._direction = np.zeros((3,))
        self._f_axis = faxis
        self._depthmax_axis = depthmaxaxis
        self._set_position(position)
        self._set_cosmin(cosmin)
        self._set_direction(direction)

    def _get_faxis(self) -> axis.EdgeAxis:
        return self._f_axis
    faxis = property(_get_faxis, None, None, 'Frequency axis object.')

    def _get_depthmaxaxis(self) -> axis.Axis:
        return self._depthmax_axis
    depthmaxaxis = property(_get_depthmaxaxis, None, None, 'Depth max axis object.')

    def _get_frequencies(self) -> np.ndarray:
        return self._f_axis.centers
    frequencies = property(_get_frequencies, None, None, 'Frequencies at which the reflectance is accumulated (rad/m).')

    def _get_depthmax(self) -> np.ndarray:
        return self._depthmax_axis.centers
    depthmax = property(_get_depthmax, None, None,
                        'Centers of the depth max axis accumulators (m).')

    def _get_depthmaxedges(self) -> np.ndarray:
        return self._depthmax_axis.edges
    depthmaxedges = property(_get_depthmaxedges, None, None,
                             'Edges of the depth max axis accumulators (m).')

    def _get_ndepthmax(self) -> int:
        return self._depthmax_axis.n
    ndepthmax = property(_get_ndepthmax, None, None,
                         'Number of accumulators in the depth max axis.')

    def _get_nf(self) -> int:
        return self._f_axis.n
    nf = property(_get_nf, None, None, 'Number of frequency accumulators.')

    def _get_position(self) -> Tuple[float, float]:
        return self._position
    def _set_position(self, value: float or Tuple[float, float]):
        self._position[:] = value
    position = property(_get_position, _set_position, None,
                       'Position of the radial frequency accumulator as a tuple (x, y).')

    def _get_cosmin(self) -> Tuple[float, float]:
        return self._cosmin
    def _set_cosmin(self, value: float or Tuple[float, float]):
        self._cosmin = min(max(float(value), 0.0), 1.0)
    cosmin = property(_get_cosmin, _set_cosmin, None,
                      'Cosine of the maximum acceptance angle.')

    def _get_direction(self) -> Tuple[float, float, float]:
        return self._direction
    def _set_direction(self, direction: Tuple[float, float, float]):
        self._direction[:] = direction
        norm = np.linalg.norm(self._direction)
        if norm == 0.0:
            raise ValueError('Direction vector norm/length must not be 0!')
        self._direction *= 1.0/norm
    direction = property(_get_direction, _set_direction, None,
                         'Detector reference direction.')

    def _get_n(self) -> int:
        return self._f_axis.n
    n = property(_get_n, None, None, 'Number of frequency accumulators (alias for nf).')

    def _get_normalized(self) -> np.ndarray:
        '''
        Net SFD reflectance as a (n_depthmax, n_f) array.
        raw has shape (n_depthmax, 2*n_f); even columns are positive,
        odd columns are absolute negative contributions.
        '''
        k = 1.0 / max(self._nphotons, 1)
        return (self.raw[:, 0::2] - self.raw[:, 1::2]) * k
    normalized = property(_get_normalized, None, None, 'Normalized.')
    reflectance = property(_get_normalized, None, None, 'Reflectance.')
    transmittance = property(_get_normalized, None, None, 'Transmittance.')

    def cl_pack(self, mc: mcobject.McObject,
                target: cltypes.Structure = None) -> cltypes.Structure:
        '''
        Fills the structure (target) with the data required by the
        Monte Carlo simulator.
        See the :py:meth:`RadialFrequencyDepthMax.cl_type` method for a
        detailed list of fields.

        Parameters
        ----------
        mc: mcobject.McObject
            Monte Carlo simulator instance.
        target: cltypes.Structure
            Ctypes structure that is filled with the source data.

        Returns
        -------
        target: cltypes.Structure
            Filled structure received as an input argument or a new
            instance if the input argument target is None.
        '''
        if target is None:
            target_type = self.cl_type(mc)
            target = target_type()

        allocation = mc.cl_allocate_rw_accumulator_buffer(self, self.shape)
        target.offset = allocation.offset

        target.position.fromarray(self._position)

        target.f_min = self._f_axis.start
        target.df = self._f_axis.step if self._f_axis.step != 0.0 else 0.0
        target.n_f = self._f_axis.n

        target.depthmax_min = self._depthmax_axis.scaled_start
        target.inv_ddepthmax = (1.0 / self._depthmax_axis.step
                                if self._depthmax_axis.step != 0.0 else 0.0)
        target.n_depthmax = self._depthmax_axis.n

        target.cos_min = self._cosmin
        target.direction.fromarray(self._direction)

        return target

    def todict(self):
        '''
        Save the accumulator configuration without the accumulator data to
        a dictionary. Use the :meth:`RadialFrequencyDepthMax.fromdict` method
        to create a new accumulator instance from the returned data.

        Returns
        -------
        data: dict
            Accumulator configuration as a dictionary.
        '''
        return {
            'type': 'RadialFrequencyDepthMax',
            'position': self._position.tolist(),
            'f_axis': self._f_axis.todict(),
            'depthmax_axis': self._depthmax_axis.todict(),
            'cosmin': self._cosmin,
            'direction': self._direction.tolist(),
        }

    @staticmethod
    def fromdict(data):
        '''
        Create an accumulator instance from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary created by the :py:meth:`RadialFrequencyDepthMax.todict` method.
        '''
        data = dict(data)
        rt_type = data.pop('type')
        if rt_type != 'RadialFrequencyDepthMax':
            raise TypeError(
                'Expected "RadialFrequencyDepthMax" type but got "{}"!'.format(rt_type))
        f_axis_data = data.pop('f_axis')
        f_axis_type = f_axis_data.pop('type')
        depthmax_axis_data = data.pop('depthmax_axis')
        depthmax_axis_type = depthmax_axis_data.pop('type')

        return RadialFrequencyDepthMax(
            getattr(axis, f_axis_type)(**f_axis_data),
            getattr(axis, depthmax_axis_type)(**depthmax_axis_data),
            **data
        )

    def __str__(self):
        return (
            'RadialFrequencyDepthMax(faxis={}, depthmaxaxis={}, '
            'position=({}, {}), cosmin={}, direction=({}, {}, {}))'.format(
                self._f_axis, self._depthmax_axis,
                *self._position, self._cosmin, *self._direction)
        )

    def __repr__(self):
        return '{} #{}'.format(self.__str__(), id(self))
