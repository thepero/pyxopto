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

from typing import Tuple, Union

import numpy as np
from xopto.mcvox.mcutil import interp

from xopto.mcvox.mcdetector.base import Detector
from xopto.mcvox import cltypes, mctypes, mcobject
from xopto.mcvox.mcutil import axis


class CartesianLens(Detector):
    @staticmethod
    def cl_type(mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClCartesianLens(cltypes.Structure):
            '''
            Structure that that represents a Cartesian lens detector
            in the Monte Carlo simulator core.

            Parameters
            ----------
            mc: McObject
                A Monte Carlo simulator instance.

            Returns
            -------
            struct: cltypes.Structure
                A structure type that represents the Cartesian lens detector in
                the Monte Carlo kernel.

            Fields
            ------
            direction_lens: mc_point3f_t
                Reference direction of the lens outward normal.
            position_lens: mc_point3f_t
                Position of the lens center.
            focal_length: mc_fp_t
                Focal length of the lens.
            f_number: mc_fp_t
                F-number of the lens.
            pos_entrance_pupil: mc_fp_t
                Position of the entrance pupil along the lens direction.
                (relative to the lens center, positive direction in front of the lens).
            pos_detector: mc_fp_t
                Position of the detector along the lens direction.
                (relative to the lens center, positive direction behind the lens).
            direction_x_det: mc_point3f_t
                Direction of the x axis of the detector plane (right direction).
            direction_y_det: mc_point3f_t
                Direction of the y axis of the detector plane (up direction).
            x_min: mc_fp_t
                The leftmost edge of the first accumulator along the x axis.
            inv_dx: mc_fp_t
                Inverse of the spacing between the accumulators along the
                x axis.
            y_min: mc_fp_t
                The leftmost edge of the first accumulator along the
                y axis.
            inv_dy: mc_fp_t
                Inverse of the spacing between the accumulators along the
                y axis.
            n_x: mc_size_t
                The number of accumulators along the x axis.
            n_y: mc_size_t
                The number of accumulators along the y axis.
            offset: mc_int_t
                The offset of the first accumulator in the Monte Carlo
                detector buffer.
            '''
            _pack_ = 1
            _fields_ = [
                ('direction_lens', T.mc_point3f_t),
                ('position_lens', T.mc_point3f_t),
                ('focal_length', T.mc_fp_t),
                ('f_number', T.mc_fp_t),
                ('pos_entrance_pupil', T.mc_fp_t),
                ('pos_detector', T.mc_fp_t),
                ('direction_x_det', T.mc_point3f_t),
                ('direction_y_det', T.mc_point3f_t),
                ('x_min', T.mc_fp_t),
                ('inv_dx', T.mc_fp_t),
                ('y_min', T.mc_fp_t),
                ('inv_dy', T.mc_fp_t),
                ('n_x', T.mc_size_t),
                ('n_y', T.mc_size_t),
                ('offset', T.mc_size_t)
            ]
        return ClCartesianLens

    def cl_declaration(self, mc: mcobject.McObject) -> str:
        '''
        Structure that defines the accumulator in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'struct MC_STRUCT_ATTRIBUTES Mc{}Detector{{'.format(Loc),
            '	mc_point3f_t direction_lens;',
            '	mc_point3f_t position_lens;',
            '	mc_fp_t focal_length;',
            '	mc_fp_t f_number;',
            '	mc_fp_t pos_entrance_pupil;',
            '	mc_fp_t pos_detector;',
            '	mc_point3f_t direction_x_det;',
            '	mc_point3f_t direction_y_det;',
            '	mc_fp_t x_min;',
            '	mc_fp_t inv_dx;',
            '	mc_fp_t y_min;',
            '	mc_fp_t inv_dy;',
            '	mc_size_t n_x;',
            '	mc_size_t n_y;',
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
            '	dbg_print("Mc{}Detector - Cartesian detector:");'.format(Loc),
            '	dbg_print_point3f(INDENT "direction_lens:", &detector->direction_lens);',
            '	dbg_print_point3f(INDENT "position_lens:", &detector->position_lens);',
            '	dbg_print_float(INDENT "focal_length:", detector->focal_length);',
            '	dbg_print_float(INDENT "f_number:", detector->f_number);',
            '	dbg_print_float(INDENT "pos_entrance_pupil:", detector->pos_entrance_pupil);',
            '	dbg_print_float(INDENT "pos_detector:", detector->pos_detector);',
            '	dbg_print_point3f(INDENT "direction_x_det:", &detector->direction_x_det);',
            '	dbg_print_point3f(INDENT "direction_y_det:", &detector->direction_y_det);',
            '	dbg_print_float(INDENT "x_min (mm):", detector->x_min*1e3f);',
            '	dbg_print_float(INDENT "inv_dx (1/mm):", detector->inv_dx*1e-3f);',
            '	dbg_print_float(INDENT "y_min (mm):", detector->y_min*1e3f);',
            '	dbg_print_float(INDENT "inv_dy (1/mm):", detector->inv_dy*1e-3f);',
            '	dbg_print_size_t(INDENT "n_x:", detector->n_x);',
            '	dbg_print_size_t(INDENT "n_y:", detector->n_y);',
            '	dbg_print_size_t(INDENT "offset:", detector->offset);',
            '};',
            '',
            'inline void mcsim_{}_detector_deposit('.format(loc),
            '		McSim *mcsim,',
            '		mc_point3f_t const *pos, mc_point3f_t const *dir,',
            '		mc_fp_t weight){',
            '',
            '	__mc_detector_mem const struct Mc{}Detector *detector = '.format(Loc),
            '		mcsim_{}_detector(mcsim);'.format(loc),
            '',
            '	__global mc_accu_t *address;',
            '	mc_int_t index_x, index_y;',
            '	mc_size_t index;',
            '',
            # Copy mc_point3f_t fields from __constant to private address space
            '   mc_point3f_t position_lens = detector->position_lens;',
            '	mc_point3f_t direction_lens = detector->direction_lens;',
            '	mc_point3f_t direction_x_det = detector->direction_x_det;',
            '	mc_point3f_t direction_y_det = detector->direction_y_det;',
            '',
            # Check if the photon is within the entrance pupil of the lens
            '   mc_point3f_t entrance_pupil_center;',
            '   mc_mad_point3f(&position_lens, &direction_lens, detector->pos_entrance_pupil, &entrance_pupil_center);',
            '   mc_point3f_t diff;',
            '   mc_mad_point3f(&entrance_pupil_center, pos, -FP_1, &diff);',
            '   mc_fp_t t_ep = mc_fdiv(mc_dot_point3f(&diff, &direction_lens), ',
            '      mc_dot_point3f(dir, &direction_lens));',
            '   mc_point3f_t intersection_ep;',
            '   mc_mad_point3f(pos, dir, t_ep, &intersection_ep);',
            '   mc_point3f_t diff_ep;',
            '   mc_mad_point3f(&intersection_ep, &entrance_pupil_center, -FP_1, &diff_ep);',
            '   if (mc_length_point3f(&diff_ep) > mc_fdiv(detector->focal_length,FP_2*detector->f_number)){',
            '       return;',
            '   }',
            '',
            # Calculate the intersection with the thin lens
            '   mc_mad_point3f(&position_lens, pos, -FP_1, &diff);',
            '   mc_fp_t t_lens = mc_fdiv(mc_dot_point3f(&diff, &direction_lens), ',
            '      mc_dot_point3f(dir, &direction_lens));',
            '   mc_point3f_t intersection_lens;',
            '   mc_mad_point3f(pos, dir, t_lens, &intersection_lens);',
            '',
            # Calculate the refracted direction dir_refracted
            '   mc_fp_t t_fp = mc_fdiv(-detector->focal_length, mc_dot_point3f(dir, &direction_lens));',
            '   mc_point3f_t focal_point;',
            '   mc_mad_point3f(&position_lens, dir, t_fp, &focal_point);',
            '   mc_point3f_t dir_refracted;',
            '   mc_mad_point3f(&focal_point, &intersection_lens, -FP_1, &dir_refracted);',
            '   mc_normalize_point3f(&dir_refracted);',
            '',
            # Calculate sensor intersection
            '   mc_point3f_t sensor_normal;',
            '   mc_cross_point3f(&direction_x_det, &direction_y_det, &sensor_normal);',
            '   mc_normalize_point3f(&sensor_normal);',
            '   mc_point3f_t sensor_center;',
            '   mc_mad_point3f(&position_lens, &direction_lens, -detector->pos_detector, &sensor_center);',
            '   mc_mad_point3f(&sensor_center, &intersection_lens, -FP_1, &diff);',
            '   mc_fp_t t_sensor = mc_fdiv(mc_dot_point3f(&diff, &sensor_normal), ',
            '      mc_dot_point3f(&dir_refracted, &sensor_normal));',
            '   mc_point3f_t intersection_sensor;',
            '   mc_mad_point3f(&intersection_lens, &dir_refracted, t_sensor, &intersection_sensor);',
            '',
            # Calculate the positions on the sensor plane
            '   mc_point3f_t sensor_rel_pos;',
            '   mc_mad_point3f(&intersection_sensor, &sensor_center, -FP_1, &sensor_rel_pos);',
            '   mc_fp_t x = mc_dot_point3f(&sensor_rel_pos, &direction_x_det);',
            '   mc_fp_t y = mc_dot_point3f(&sensor_rel_pos, &direction_y_det);',
            '',
            '	dbg_print_status(mcsim, "{} CartesianLens detector hit");'.format(Loc),
            '',
            '	index_x = mc_int((x - detector->x_min)*detector->inv_dx);',
            '	index_x = mc_clip(index_x, 0, detector->n_x - 1);',
            '',
            '	index_y = mc_int((y - detector->y_min)*detector->inv_dy);',
            '	index_y = mc_clip(index_y, 0, detector->n_y - 1);',
            '',
            '	index = index_y*detector->n_x + index_x;',
            '',
            '	address = mcsim_accumulator_buffer_ex(',
            '		mcsim, index + detector->offset);',
            '',
            '	uint32_t ui32w = weight_to_int(weight);',
            '',
            '	if (ui32w > 0){',
            '		dbg_print_uint("{} CartesianLens detector depositing int:", ui32w);'.format(Loc),
            '		accumulator_deposit(address, ui32w);',
            '	};',
            '};'
        ))

    def __init__(self, xaxis: Union[axis.Axis, 'CartesianLens'],
                yaxis: axis.Axis = None,
                direction_lens: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                position_lens: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                focal_length: float = 1.0, f_number: float = 1.0,
                pos_entrance_pupil: float = 0.0, pos_detector: float = 0.0,
                direction_x_det: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                direction_y_det: Tuple[float, float, float] = (0.0, 1.0, 0.0)):
        '''
        2D Cartesian lens reflectance/transmittance accumulator in the x-y plane.

        The grid of the Cartesian lens accumulators corresponds to a 2D numpy array
        with the first dimension representing the y axis and second dimension
        representing the x axis (reflectance[y, x] or transmittance[y, x]).

        Parameters
        ----------
        xaxis: axis.Axis or CartesianLens
            Object that defines accumulators along the x axis.
        yaxis: axis.Axis
            Object that defines accumulators along the y axis. If None, the
            y axis will equal x axis.
        direction_lens: Tuple[float, float, float]
            Cosine of the maximum acceptance angle (relative to the direction)
            of the detector.
        position_lens: Tuple[float, float, float]
            Position of the lens center.
        focal_length: float
            Focal length of the lens.
        f_number: float
            F-number of the lens.
        pos_entrance_pupil: float
            Position of the entrance pupil along the lens direction (relative to
            the lens center, positive direction in front of the lens).
        pos_detector: float
            Position of the detector along the lens direction (relative to
            the lens center, positive direction behind the lens).
        direction_x_det: Tuple[float, float, float]
            Direction of the x axis of the detector plane (right direction).
        direction_y_det: Tuple[float, float, float]
            Direction of the y axis of the detector plane (up direction).
        '''
        if isinstance(xaxis, CartesianLens):
            detector = xaxis
            xaxis = type(detector.xaxis)(detector.xaxis)
            yaxis = type(detector.yaxis)(detector.yaxis)
            direction_lens = detector.direction_lens
            position_lens = detector.position_lens
            focal_length = detector.focal_length
            f_number = detector.f_number
            pos_entrance_pupil = detector.pos_entrance_pupil
            pos_detector = detector.pos_detector
            direction_x_det = detector.direction_x_det
            direction_y_det = detector.direction_y_det
            raw_data = np.copy(detector.raw)
            nphotons = detector.nphotons
        else:
            if yaxis is None:
                yaxis = axis.Axis(xaxis)

            raw_data = np.zeros((yaxis.n, xaxis.n))
            nphotons = 0

        super().__init__(raw_data, nphotons)

        self._direction_lens = np.zeros((3,))
        self._position_lens = np.zeros((3,))
        self._focal_length = 0.0
        self._f_number = 0.0
        self._pos_entrance_pupil = 0.0
        self._pos_detector = 0.0
        self._direction_x_det = np.zeros((3,))
        self._direction_y_det = np.zeros((3,))
        self._x_axis = xaxis
        self._y_axis = yaxis

        self._set_direction_lens(direction_lens)
        self._set_position_lens(position_lens)
        self._set_focal_length(focal_length)
        self._set_f_number(f_number)
        self._set_pos_entrance_pupil(pos_entrance_pupil)
        self._set_pos_detector(pos_detector)
        self._set_direction_x_det(direction_x_det)
        self._set_direction_y_det(direction_y_det)

        self._accumulators_area = self._x_axis.step*self._y_axis.step

    def _get_x_axis(self) -> axis.Axis:
        return self._x_axis
    xaxis = property(_get_x_axis, None, None, 'Axis object of the x axis.')

    def _get_y_axis(self) -> axis.Axis:
        return self._y_axis
    yaxis = property(_get_y_axis, None, None, 'Axis object of the y axis.')
    
    def _get_focal_length(self) -> float:
        return self._focal_length
    def _set_focal_length(self, value: float):
        self._focal_length = float(value)
    focal_length = property(_get_focal_length, _set_focal_length, None,
                            'Focal length of the lens.')
    
    def _get_f_number(self) -> float:
        return self._f_number
    def _set_f_number(self, value: float):
        self._f_number = float(value)
    f_number = property(_get_f_number, _set_f_number, None,
                        'F-number of the lens.')
    
    def _get_pos_entrance_pupil(self) -> float:
        return self._pos_entrance_pupil
    def _set_pos_entrance_pupil(self, value: float):
        self._pos_entrance_pupil = float(value)
    pos_entrance_pupil = property(_get_pos_entrance_pupil, _set_pos_entrance_pupil, None,
                                 'Position of the entrance pupil along the lens direction '
                                 '(relative to the lens center, '
                                 'positive direction in front of the lens).')
    
    def _get_pos_detector(self) -> float:
        return self._pos_detector
    def _set_pos_detector(self, value: float):
        self._pos_detector = float(value)
    pos_detector = property(_get_pos_detector, _set_pos_detector, None,
                            'Position of the detector along the lens direction '
                            '(relative to the lens center, '
                            'positive direction in front of the lens).')

    def _get_direction_lens(self) -> Tuple[float, float, float]:
        return self._direction_lens
    def _set_direction_lens(self, direction_lens: Tuple[float, float, float]):
        self._direction_lens[:] = direction_lens
        norm = np.linalg.norm(self._direction_lens)
        if norm == 0.0:
            raise ValueError('Direction vector norm/length must not be 0!')
        self._direction_lens *= 1.0/norm
    direction_lens = property(_get_direction_lens, _set_direction_lens, None,
                        'Detector reference direction.')
    
    def _get_position_lens(self) -> Tuple[float, float, float]:
        return self._position_lens
    def _set_position_lens(self, position_lens: Tuple[float, float, float]):
        self._position_lens[:] = position_lens
    position_lens = property(_get_position_lens, _set_position_lens, None,
                        'Position of the lens.')
    
    def _get_direction_x_det(self) -> Tuple[float, float, float]:
        return self._direction_x_det
    def _set_direction_x_det(self, direction_x_det: Tuple[float, float, float]):
        self._direction_x_det[:] = direction_x_det
        norm = np.linalg.norm(self._direction_x_det)
        if norm == 0.0:
            raise ValueError('Direction vector x_det norm/length must not be 0!')
        self._direction_x_det *= 1.0/norm
    direction_x_det = property(_get_direction_x_det, _set_direction_x_det, None,
                        'Detector reference direction along x axis.')
    
    def _get_direction_y_det(self) -> Tuple[float, float, float]:
        return self._direction_y_det
    def _set_direction_y_det(self, direction_y_det: Tuple[float, float, float]):
        self._direction_y_det[:] = direction_y_det
        norm = np.linalg.norm(self._direction_y_det)
        if norm == 0.0:
            raise ValueError('Direction vector y_det norm/length must not be 0!')
        self._direction_y_det *= 1.0/norm
    direction_y_det = property(_get_direction_y_det, _set_direction_y_det, None,
                        'Detector reference direction along y axis.')

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

        target.direction_lens.fromarray(self._direction_lens)
        target.position_lens.fromarray(self._position_lens)

        target.focal_length = self._focal_length
        target.f_number = self._f_number
        target.pos_entrance_pupil = self._pos_entrance_pupil
        target.pos_detector = self._pos_detector
        target.direction_x_det.fromarray(self._direction_x_det)
        target.direction_y_det.fromarray(self._direction_y_det)

        target.x_min = self._x_axis.start
        if self._x_axis.n > 1:
            target.inv_dx = 1.0/self._x_axis.step
        else:
            target.inv_dx = 0.0
        target.y_min = self._y_axis.start
        if self._y_axis.n > 1:
            target.inv_dy = 1.0/self._y_axis.step
        else:
            target.inv_dy = 0.0

        target.n_x = self._x_axis.n
        target.n_y = self._y_axis.n

        return target

    def todict(self) -> dict:
        '''
        Save the accumulator configuration without the accumulator data to
        a dictionary. Use the :meth:`Cartesian.fromdict` method to create a new
        accumulator instance from the returned data.

        Returns
        -------
        data: dict
            Accumulator configuration as a dictionary.
        '''
        return {
            'type':'CartesianLens',
            'x_axis': self._x_axis.todict(),
            'y_axis': self._y_axis.todict(),
            'direction_lens': self._direction_lens.tolist(),
            'position_lens': self._position_lens.tolist(),
            'focal_length': self._focal_length,
            'f_number': self._f_number,
            'pos_entrance_pupil': self._pos_entrance_pupil,
            'pos_detector': self._pos_detector,
            'direction_x_det': self._direction_x_det.tolist(),
            'direction_y_det': self._direction_y_det.tolist()
        }

    @staticmethod
    def fromdict(data: dict) -> 'CartesianLens':
        '''
        Create an accumulator instance from a dictionary.

        Parameters
        ----------
        data: dict
            Dictionary created by the py:meth:`CartesianLens.todict` method.
        '''
        data_ = dict(data)
        detector_type = data_.pop('type')
        if detector_type != 'CartesianLens':
            raise TypeError('Expected "CartesianLens" type but got "{}"!'.format(
                detector_type))
        xaxis_data = data_.pop('x_axis')
        xaxis_type = xaxis_data.pop('type')

        yaxis_data = data_.pop('y_axis')
        yaxis_type = yaxis_data.pop('type')

        return CartesianLens(
            getattr(axis, xaxis_type)(**xaxis_data),
            getattr(axis, yaxis_type)(**yaxis_data),
            **data_
        )

    def plot(self, scale: str = 'log', raw: bool = False, show: bool = True):
        '''
        Show the detector contet as a 2D image.

        Parameters
        ----------
        scale: str
            Data scaling can be "log" for logarithmic or "lin" for linear.
        raw: bool
            Set to True to show the raw data. Default is False and shows the
            normalized (reflectance) content.
        show: bool 
        '''
        import matplotlib.pyplot as pp

        extent = [self._x_axis.start, self._x_axis.stop,
                  self._y_axis.start, self._y_axis.stop]

        data = self.raw if raw else self.reflectance
        which = 'raw' if raw else 'reflectance'
        
        if scale == 'log':
            mask = data > 0.0
            if mask.size > 0:
                log_data = np.tile(np.log10(data[mask].min()), data.shape)
                log_data[mask] = np.log10(data[mask])
                data = log_data

        fig, ax = pp.subplots()
        img = ax.imshow(data, extent=extent, origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        pp.colorbar(img)

        fig.canvas.manager.set_window_title(
            'CartesianLens detector - {} - {}'.format(scale, which))

        if show:
            pp.show()

    def __str__(self):
        return 'CartesianLens(xaxis={}, yaxis={}, direction_lens={}, '\
               'position_lens={}, focal_length={}, f_number={}, '\
               'pos_entrance_pupil={}, pos_detector={}, '\
               'direction_x_det={}, direction_y_det={})'.format(
                   self._x_axis, self._y_axis, self._direction_lens,
                   self._position_lens, self._focal_length, self._f_number,
                   self._pos_entrance_pupil, self._pos_detector,
                   self._direction_x_det, self._direction_y_det)

    def __repr__(self):
        return '{} #{}'.format(self.__str__(), id(self))
