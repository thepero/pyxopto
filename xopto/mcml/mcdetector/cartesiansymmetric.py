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
from xopto.mcml import cltypes, mcobject
from xopto.mcml.mcutil import axis


class CartesianSymmetricX(Detector):
    @staticmethod
    def cl_type(mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClCartesianSymmetricX(cltypes.Structure):
            '''
            Structure that that represents a Cartesian detector
            in the Monte Carlo simulator core (linear or log)

            Fields
            ------
            direction: mc_point3f_t
                Reference direction/orientation of the detector.
            position_x: mc_fp_t
                Coordinate x of the origin/center along the x-axis.
            x_offset: mc_fp_t
                Offset of the first accumulator relative to the axis origin
                defined by the center parameter.
            inv_dx: mc_fp_t
                Inverse value of the spacing between the accumulators.
            y_min: mc_fp_t
                The leftmost edge of the first accumulator along the
                y axis.
            inv_dy: mc_fp_t
                Inverse of the spacing between the accumulators along the
                y axis.
            cos_min: mc_fp_t
                Cosine of the maximum acceptance angle (relative to the
                reference direction of the detector).
            n_half_x: mc_size_t
                The number of accumulators in the positive or negative
                direction along the x axis from the center of the detector.
            n_y: mc_size_t
                The number of accumulators along the y axis.
            log_scale_x: mc_int_t
                A flag indicating logarithmic spacing of the accumulators
                along the x axis.
            offset: mc_size_t
                The offset of the first accumulator in the Monte Carlo
                detector buffer.

            Note
            ----
            Note that for logarithmic accumulators along x
            the value of inv_dx is passed in a logarithmic scale.
            '''
            _pack_ = 1
            _fields_ = [
                ('direction', T.mc_point3f_t),
                ('position_x', T.mc_fp_t),
                ('x_offset', T.mc_fp_t),
                ('inv_dx', T.mc_fp_t),
                ('y_min', T.mc_fp_t),
                ('inv_dy', T.mc_fp_t),
                ('cos_min', T.mc_fp_t),
                ('n_half_x', T.mc_size_t),
                ('n_y', T.mc_size_t),
                ('log_scale_x', T.mc_int_t),
                ('offset', T.mc_size_t),
            ]
        return ClCartesianSymmetricX

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
            '	mc_fp_t inv_dx;',
            '	mc_fp_t y_min;',
            '	mc_fp_t inv_dy;',
            '	mc_fp_t cos_min;',
            '	mc_size_t n_half_x;',
            '	mc_size_t n_y;',
            '	mc_int_t log_scale_x;',
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
            '	dbg_print("Mc{}Detector - CartesianSymmetricX detector:");'.format(Loc),
            '	dbg_print_point3f(INDENT "direction:", &detector->direction);',
            '	dbg_print_float(INDENT "position_x (mm):", detector->position_x*1e3f);',
            '	dbg_print_float(INDENT "x_offset (mm):", detector->x_offset*1e3f);',
            '	dbg_print_float(INDENT "inv_dx (1/mm):", detector->inv_dx*1e-3f);',
            '	dbg_print_float(INDENT "y_min (mm):", detector->y_min*1e3f);',
            '	dbg_print_float(INDENT "inv_dy (1/mm):", detector->inv_dy*1e-3f);',
            '	dbg_print_float(INDENT "cos_min:", detector->cos_min);',
            '	dbg_print_size_t(INDENT "n_half_x:", detector->n_half_x);',
            '	dbg_print_size_t(INDENT "n_y:", detector->n_y);',
            '	dbg_print_int(INDENT "log_scale_x:", detector->log_scale_x);',
            '	dbg_print_size_t(INDENT "offset:", detector->offset);',
            '};',
            '',
            'inline void mcsim_{}_detector_deposit('.format(loc),
            '		McSim *mcsim,',
            '		mc_point3f_t const *pos, mc_point3f_t const *dir,',
            '		mc_fp_t weight){',
            '',
            '	__global mc_accu_t *address;',
            '	__mc_detector_mem const struct Mc{}Detector *detector = '.format(Loc),
            '		mcsim_{}_detector(mcsim);'.format(loc),
            '',
            '	mc_fp_t x = mc_fabs(pos->x - detector->position_x);',
            '	mc_int_t index_y;',
            '	mc_size_t accu_index;',
            '',
            '	dbg_print_status(mcsim, "{} CartesianSymmetricX detector hit");'.format(Loc),
            '',
            '	if (detector->log_scale_x)',
            '		x = mc_log(mc_fmax(x, FP_RMIN));',
            '	mc_int_t index_x = mc_int((x - detector->x_offset)*detector->inv_dx);',
            '	index_x = mc_clip(index_x, 0, detector->n_half_x - 1);',
            '	index_x = (pos->x - detector->position_x >= FP_0) ?',
            '		index_x + detector->n_half_x : detector->n_half_x - index_x - 1;',
            '',
            '	index_y = mc_int((pos->y - detector->y_min)*detector->inv_dy);',
            '	index_y = mc_clip(index_y, 0, detector->n_y - 1);',
            '',
            '	accu_index = index_y*(2*detector->n_half_x) + index_x;',
            '',
            '	address = mcsim_accumulator_buffer_ex(',
            '		mcsim, detector->offset + accu_index);',
            '',
            '	mc_point3f_t detector_direction = detector->direction;',
            '	uint32_t ui32w = weight_to_int(weight)*',
            '		(detector->cos_min <= mc_fabs(mc_dot_point3f(dir, &detector_direction)));',
            '',
            '	if (ui32w > 0){',
            '		dbg_print_uint("{} CartesianSymmetricX detector depositing int:", ui32w);'.format(Loc),
            '		accumulator_deposit(address, ui32w);',
            '	};',
            '};'
        ))

    def __init__(self, xaxis, yaxis=None, cosmin=0.0,
                 direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        '''
        2D Cartesian reflectance/transmittance accumulator in the x-y plane with
        symmetric detector across the x axis and stadnard detector for y axis.

        The grid of the Cartesian accumulators corresponds to a 2D numpy array
        with the first dimension representing the y axis and second dimension
        representing the x axis (reflectance[y, x] or transmittance[y, x]).

        Parameters
        ----------
        xaxis: axis.SymmetricAxis
            Object that defines the accumulators along the x axis
            (this axis supports log-scale).
        yaxis: axis.Axis
            Object that defines accumulators along the y axis. If None, the
            y axis will equal x axis.
        cosmin: float
            Cosine of the maximum acceptance angle (relative to the direction)
            of the detector.
         direction: (float, float, float)
            Reference direction/orientation of the source.
        '''
        if isinstance(xaxis, CartesianSymmetricX):
            detector = xaxis
            xaxis = type(detector.xaxis)(detector.xaxis)
            yaxis = type(detector.yaxis)(detector.yaxis)
            cosmin = detector.cosmin
            direction = detector.direction
            raw_data = np.copy(detector.raw)
            nphotons = detector.nphotons
        else:
            if yaxis is None:
                yaxis = axis.Axis(xaxis)

            raw_data = np.zeros((yaxis.n, xaxis.n))
            nphotons = 0

        super().__init__(raw_data, nphotons)
        self._cosmin = 0.0
        self._direction = np.zeros((3,))

        self._x_axis = xaxis
        self._dx = self._x_axis.edges[1:] - self._x_axis.edges[:-1]
        self._y_axis = yaxis
        self._dy = self._y_axis.edges[1:] - self._y_axis.edges[:-1]

        self._set_cosmin(cosmin)
        self._set_direction(direction)

        self._accumulators_area = self._dx[np.newaxis, :] * self._dy[:, np.newaxis]

    def _get_x_axis(self) -> axis.SymmetricAxis:
        return self._x_axis
    xaxis = property(_get_x_axis, None, None, 'Axis object of the x axis.')

    def _get_y_axis(self) -> axis.Axis:
        return self._y_axis
    yaxis = property(_get_y_axis, None, None, 'Axis object of the y axis.')

    def _get_cosmin(self) -> float:
        return self._cosmin
    def _set_cosmin(self, value: float):
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
    x = property(_get_x, None, None,
                 'Centers of the accumulators along the x axis.')

    def _get_y(self) -> np.ndarray:
        return self._y_axis.centers
    y = property(_get_y, None, None,
                 'Centers of the accumulators along the y axis.')

    def _get_x_edges(self) -> np.ndarray:
        return self._x_axis.edges
    xedges = property(_get_x_edges, None, None,
                      'Edges of the accumulators along the x axis.')

    def _get_y_edges(self) -> np.ndarray:
        return self._y_axis.edges
    yedges = property(_get_y_edges, None, None,
                      'Edges of the accumulators along the y axis.')

    def _get_nx(self) -> int:
        return self._x_axis.n
    nx = property(_get_nx, None, None,
                  'Number of accumulators along the x axis.')

    def _get_ny(self) -> int:
        return self._y_axis.n
    ny = property(_get_ny, None, None,
                  'Number of accumulators along the y axis.')
    
    def _get_logscale(self) -> bool:
        return self._x_axis.logscale
    logscale_x = property(_get_logscale, None, None, 'Axis log scale.')

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Returns 2D arrays of x and y coordinates of the centers of accumulators
        that match the size of the reflectance / transmittance arrays.
        The grid of the Cartesian accumulators corresponds to a 2D numpy array
        with the first dimension representing the y axis and second dimension
        representing the x axis (reflectance[y, x] or transmittance[y, x]).

        Returns
        -------
        x: np.ndarray
            A 2D array of x coordinates.
        y: np.ndarray
            A 2D array of y coordinates.
        '''
        Y, X = np.meshgrid(self.y, self.x, indexing='ij')
        return X, Y

    def _get_normalized(self) -> np.ndarray:
        return self.raw*(1.0/(max(self.nphotons, 1.0)*self._accumulators_area))
    normalized = property(_get_normalized, None, None, 'Normalized.')
    reflectance = property(_get_normalized, None, None, 'Reflectance.')
    transmittance = property(_get_normalized, None, None, 'Transmittance.')

    def cl_pack(self, mc: mcobject.McObject, target: cltypes.Structure = None) \
            -> cltypes.Structure:
        '''
        Fills the structure (target) with the data required by the
        Monte Carlo simulator.
        See the :py:meth:`Cartesian.cl_type` method for a detailed
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
            target.inv_dx = 1.0/self._x_axis.step
        else:
            target.inv_dx = 0.0
        target.y_min = self._y_axis.start
        if self._y_axis.n > 1:
            target.inv_dy = 1.0/self._y_axis.step
        else:
            target.inv_dy = 0.0
        target.cos_min = self.cosmin
        target.n_half_x = self._x_axis.n_half
        target.n_y = self._y_axis.n
        target.log_scale_x = self._x_axis.logscale

        return target

    def todict(self) -> dict:
        '''
        Save the accumulator configuration without the accumulator data to
        a dictionary. Use the :meth:`CartesianSymmetricX.fromdict` method to create a new
        accumulator instance from the returned data.

        Returns
        -------
        data: dict
            Accumulator configuration as a dictionary.
        '''
        return {
            'type':'CartesianSymmetricX',
            'x_axis': self._x_axis.todict(),
            'y_axis': self._y_axis.todict(),
            'cosmin': self._cosmin,
            'direction': self._direction.tolist()
        }

    @staticmethod
    def fromdict(data: dict) -> 'CartesianSymmetricX':
        '''
        Create an accumulator instance from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary created by the py:meth:`CartesianSymmetricX.todict` method.
        '''
        data_ = dict(data)
        detector_type = data_.pop('type')
        if detector_type != 'CartesianSymmetricX':
            raise TypeError('Expected "CartesianSymmetricX" type but got "{}"!'.format(
                detector_type))
        xaxis_data = data_.pop('x_axis')
        xaxis_type = xaxis_data.pop('type')

        yaxis_data = data_.pop('y_axis')
        yaxis_type = yaxis_data.pop('type')

        return CartesianSymmetricX(
            getattr(axis, xaxis_type)(**xaxis_data),
            getattr(axis, yaxis_type)(**yaxis_data),
            **data_
        )

    def __str__(self):
        return 'CartesianSymmetricX(xaxis={}, yaxis={}, cosmin={}, '\
               'direction=({}, {}, {}))'.format(
                   self._x_axis, self._y_axis, self._cosmin, *self._direction)

    def __repr__(self):
        return '{} #{}'.format(self.__str__(), id(self))
