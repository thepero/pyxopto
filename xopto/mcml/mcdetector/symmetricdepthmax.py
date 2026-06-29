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


class SymmetricXDepthMax(Detector):
    @staticmethod
    def cl_type(mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClSymmetricXDepthMax(cltypes.Structure):
            '''
            Structure that represents a symmetric x detector with maximum
            depth tracking in the Monte Carlo simulator core.

            Fields
            ------
            direction: mc_point3f_t
                Reference direction/orientation of the detector.
            position_x: mc_fp_t
                Coordinate x of the origin/center of the symmetric detector.
            x_offset: mc_fp_t
                Offset of the first accumulator relative to the axis origin
                defined by the center parameter.
            inv_step: mc_fp_t
                Inverse value of the spacing between the accumulators.
            depthmax_min: mc_fp_t
                The leftmost edge of the first maximum depth accumulator.
            inv_ddepthmax: mc_fp_t
                Inverse of the width of the maximum depth accumulators.
            cos_min: mc_fp_t
                Cosine of the maximum acceptance angle (relative to the
                reference direction of the detector).
            n_half: mc_size_t
                The number of accumulators in the positive or negative
                direction along the x axis from the center of the detector.
            n_depthmax: mc_size_t
                The number of maximum depth accumulators.
            log_scale: mc_int_t
                A flag indicating logarithmic spacing of the accumulators
                along the x axis.
            offset: mc_size_t
                The offset of the first accumulator in the Monte Carlo
                detector buffer.

            Note
            ----
            Note that for logarithmic accumulators the value of
            inv_step is passed in a logarithmic scale.
            '''
            _pack_ = 1
            _fields_ = [
                ('direction', T.mc_point3f_t),
                ('position_x', T.mc_fp_t),
                ('x_offset', T.mc_fp_t),
                ('inv_step', T.mc_fp_t),
                ('depthmax_min', T.mc_fp_t),
                ('inv_ddepthmax', T.mc_fp_t),
                ('cos_min', T.mc_fp_t),
                ('n_half', T.mc_size_t),
                ('n_depthmax', T.mc_size_t),
                ('log_scale', T.mc_int_t),
                ('offset', T.mc_size_t),
            ]
        return ClSymmetricXDepthMax

    def cl_declaration(self, mc: mcobject.McObject) -> str:
        '''
        Structure that defines the accumulator in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'struct MC_STRUCT_ATTRIBUTES Mc{}Detector{{'.format(Loc),
            '	mc_point3f_t direction;',
            '	mc_fp_t position_x;',
            '	mc_fp_t x_offset;',
            '	mc_fp_t inv_step;',
            '	mc_fp_t depthmax_min;',
            '	mc_fp_t inv_ddepthmax;',
            '	mc_fp_t cos_min;',
            '	mc_size_t n_half;',
            '	mc_size_t n_depthmax;',
            '	mc_int_t log_scale;',
            '	mc_size_t offset;',
            '};'
        ))

    def cl_implementation(self, mc: mcobject.McObject) -> str:
        '''
        Implementation of the accumulator in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'void dbg_print_{}_detector('.format(loc),
            '		__mc_detector_mem const Mc{}Detector *detector){{'.format(Loc),
            '	dbg_print("Mc{}Detector - SymmetricXDepthMax detector:");'.format(Loc),
            '	dbg_print_point3f(INDENT "direction:", &detector->direction);',
            '	dbg_print_float(INDENT "position_x (mm):", detector->position_x*1e3f);',
            '	dbg_print_float(INDENT "x_offset (mm):", detector->x_offset*1e3f);',
            '	dbg_print_float(INDENT "inv_step (1/mm):", detector->inv_step*1e-3f);',
            '	dbg_print_float(INDENT "depthmax_min (um):", detector->depthmax_min*1e6f);',
            '	dbg_print_float(INDENT "inv_ddepthmax (1/um)", detector->inv_ddepthmax*1e-6f);',
            '	dbg_print_float(INDENT "cos_min:", detector->cos_min);',
            '	dbg_print_size_t(INDENT "n_half:", detector->n_half);',
            '	dbg_print_size_t(INDENT "n_depthmax:", detector->n_depthmax);',
            '	dbg_print_int(INDENT "log_scale:", detector->log_scale);',
            '	dbg_print_size_t(INDENT "offset:", detector->offset);',
            '};',
            '',
            'inline void mcsim_{}_detector_deposit('.format(loc),
            '		McSim *mcsim, ',
            '		mc_point3f_t const *pos, mc_point3f_t const *dir,',
            '		mc_fp_t weight){',
            '',
            '	__global mc_accu_t *address;',
            '	__mc_detector_mem const struct Mc{}Detector *detector = '.format(Loc),
            '		mcsim_{}_detector(mcsim);'.format(loc),
            ''
            '	mc_fp_t x = mc_fabs(pos->x - detector->position_x);',
            '',
            '	dbg_print_status(mcsim, "{} SymmetricXDepthMax detector hit");'.format(Loc),
            '',
            '	if (detector->log_scale)',
            '		x = mc_log(mc_fmax(x, FP_RMIN));',
            '',
            '	mc_int_t index_x = mc_int((x - detector->x_offset)*detector->inv_step);',
            '	index_x = mc_clip(index_x, 0, detector->n_half - 1);',
            '	mc_size_t accu_index = (mcsim_position_x(mcsim) - detector->position_x >= FP_0) ?',
            '		index_x + detector->n_half : detector->n_half - index_x - 1;',
            '',
            '	mc_fp_t depth = mcsim_depth_max(mcsim);',
            '	mc_int_t depth_index = mc_int((depth - detector->depthmax_min)*detector->inv_ddepthmax);',
            '	depth_index = mc_clip(depth_index, 0, detector->n_depthmax - 1);',
            '',
            '	mc_size_t index = depth_index*(detector->n_half*2) + accu_index;',
            '',
            '	address = mcsim_accumulator_buffer_ex(',
            '		mcsim, detector->offset + index);',
            '',
            '	mc_point3f_t detector_direction = detector->direction;',
            '	uint32_t ui32w = weight_to_int(weight)*',
            '		(detector->cos_min <= mc_fabs(mc_dot_point3f(dir, &detector_direction)));',
            '',
            '	if (ui32w > 0){',
            '		dbg_print_uint("{} SymmetricXDepthMax detector depositing int:", ui32w);'.format(Loc),
            '		accumulator_deposit(address, ui32w);',
            '	};',
            '};'
        ))

    def cl_options(self, mc, target=None) -> mcoptions.RawOptions:
        '''
        OpenCL kernel options defined by this object.
        '''
        return [('MC_TRACK_DEPTH_MAX', True)]

    def __init__(self, xaxis: axis.SymmetricAxis,
                 depthmaxaxis: axis.Axis = None,
                 cosmin: float = 0.0,
                 direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        '''
        Symmetric x depth-max reflectance-transmittance accumulator.

        Parameters
        ----------
        xaxis: axis.SymmetricAxis
            Object that defines the accumulators along the x axis
            (this axis supports log-scale).
        depthmaxaxis: axis.Axis
            Object that defines the accumulators along the depth max axis.
        cosmin: float
            Cosine of the maximum acceptance angle (relative to the
            reference direction of the detector).
        direction: (float, float, float)
            Reference direction/orientation of the detector.

        Note
        ----
        The first dimension of the accumulator represents the depth max axis,
        the second dimension represents the x axis.
        '''
        if isinstance(xaxis, SymmetricXDepthMax):
            sxdm = xaxis
            xaxis = type(sxdm.xaxis)(sxdm.xaxis)
            depthmaxaxis = type(sxdm.depthmaxaxis)(sxdm.depthmaxaxis)
            cosmin = sxdm.cosmin
            direction = sxdm.direction
            raw_data = np.copy(sxdm.raw)
            nphotons = sxdm.nphotons
        else:
            if depthmaxaxis is None:
                depthmaxaxis = axis.Axis(0.0, 1.0, 1)

            raw_data = np.zeros((depthmaxaxis.n, xaxis.n))
            nphotons = 0

        super().__init__(raw_data, nphotons)

        self._x_axis = xaxis
        self._depthmax_axis = depthmaxaxis

        self._inv_dx = 1.0/(self._x_axis.edges[1:] - self._x_axis.edges[:-1])
        self._inv_dx.shape = (1, self._inv_dx.size)

        self._cosmin = 0.
        self._direction = np.zeros((3,))
        self._set_cosmin(cosmin)
        self._set_direction(direction)

    def _get_xaxis(self) -> axis.SymmetricAxis:
        return self._x_axis
    xaxis = property(_get_xaxis, None, None, 'Axis object.')

    def _get_depthmaxaxis(self) -> axis.Axis:
        return self._depthmax_axis
    depthmaxaxis = property(_get_depthmaxaxis, None, None, 'Depth max axis object.')

    def _get_cosmin(self) -> float:
        return self._cosmin
    def _set_cosmin(self, value):
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

    def _get_x(self) -> np.ndarray:
        return self._x_axis.centers
    x = property(_get_x, None, None, 'Centers of the accumulators.')

    def _get_edges(self) -> np.ndarray:
        return self._x_axis.edges
    edges = property(_get_edges, None, None, 'Edges of the accumulators.')

    def _get_n(self) -> int:
        return self._x_axis.n
    n = property(_get_n, None, None, 'Number of accumulators along the x axis.')

    def _get_logscale(self) -> bool:
        return self._x_axis.logscale
    logscale = property(_get_logscale, None, None, 'Axis log scale.')

    def _get_depthmax(self) -> np.ndarray:
        return self._depthmax_axis.centers
    depthmax = property(_get_depthmax, None, None,
                        'Centers of the depth max axis accumulators.')

    def _get_depthmaxedges(self) -> np.ndarray:
        return self._depthmax_axis.edges
    depthmaxedges = property(_get_depthmaxedges, None, None,
                             'Edges of the depth max axis accumulators.')

    def _get_ndepthmax(self) -> int:
        return self._depthmax_axis.n
    ndepthmax = property(_get_ndepthmax, None, None,
                         'Number of accumulators in the depth max axis.')

    def _get_normalized(self) -> np.ndarray:
        return self.raw * self._inv_dx * (1.0 / max(self.nphotons, 1.0))
    normalized = property(_get_normalized, None, None, 'Normalized.')
    reflectance = property(_get_normalized, None, None, 'Reflectance.')
    transmittance = property(_get_normalized, None, None, 'Transmittance.')

    def cl_pack(self, mc: mcobject.McObject,
                target: cltypes.Structure = None) -> cltypes.Structure:
        '''
        Fills the structure (target) with the data required by the
        Monte Carlo simulator.
        See the :py:meth:`SymmetricXDepthMax.cl_type` method for a detailed
        list of fields.

        Parameters
        ----------
        mc: mcobject.McObject
            Monte Carlo simulator instance.
        target: cltypes.Structure
            Ctypes structure that is filled with the source data.

        Returns
        -------
        target: cltypes.Structure
            Filled ctypes structure received as an input argument or a new
            instance if the input argument target is None.
        '''
        if target is None:
            target_type = self.cl_type(mc)
            target = target_type()

        allocation = mc.cl_allocate_rw_accumulator_buffer(self, self.shape)
        target.offset = allocation.offset

        target.direction.fromarray(self._direction)

        target.position_x = self._x_axis.center
        target.x_offset = self._x_axis.scaled_offset
        if self._x_axis.step != 0.0:
            target.inv_step = 1.0/self._x_axis.step
        else:
            target.inv_step = 0.0
        target.log_scale = self._x_axis.logscale
        target.n_half = self._x_axis.n_half

        target.depthmax_min = self._depthmax_axis.scaled_start
        if self._depthmax_axis.step != 0.0:
            target.inv_ddepthmax = 1.0/self._depthmax_axis.step
        else:
            target.inv_ddepthmax = 0.0
        target.n_depthmax = self._depthmax_axis.n

        target.cos_min = self._cosmin

        return target

    def todict(self) -> dict:
        '''
        Save the accumulator configuration without the accumulator data to
        a dictionary. Use the :py:meth:`SymmetricXDepthMax.fromdict` method to
        create a new accumulator instance from the returned data.

        Returns
        -------
        data: dict
            Accumulator configuration as a dictionary.
        '''
        return {
            'type': 'SymmetricXDepthMax',
            'x_axis': self._x_axis.todict(),
            'depthmax_axis': self._depthmax_axis.todict(),
            'cosmin': self._cosmin,
            'direction': self._direction.tolist()
        }

    @staticmethod
    def fromdict(data: dict) -> 'SymmetricXDepthMax':
        '''
        Create an accumulator instance from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary created by the :py:meth:`SymmetricXDepthMax.todict` method.
        '''
        data = dict(data)
        detector_type = data.pop('type')
        if detector_type != 'SymmetricXDepthMax':
            raise TypeError(
                'Expected "SymmetricXDepthMax" type but got "{}"!'.format(
                    detector_type))
        x_axis_data = data.pop('x_axis')
        x_axis_type = x_axis_data.pop('type')
        depthmax_axis_data = data.pop('depthmax_axis')
        depthmax_axis_type = depthmax_axis_data.pop('type')
        return SymmetricXDepthMax(
            getattr(axis, x_axis_type)(**x_axis_data),
            getattr(axis, depthmax_axis_type)(**depthmax_axis_data),
            **data
        )

    def __str__(self):
        return 'SymmetricXDepthMax(xaxis={}, depthmaxaxis={}, cosmin={}, '\
               'direction=({}, {}, {}))'.format(
                   self._x_axis, self._depthmax_axis, self._cosmin,
                   *self._direction)

    def __repr__(self):
        return '{} #{}'.format(self.__str__(), id(self))
