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
from xopto.mcml import cltypes, mctypes, mcobject
from xopto.mcml.mcutil import axis


class XFrequency(Detector):
    @staticmethod
    def cl_type(mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClXFrequency(cltypes.Structure):
            '''
            Structure that represents a detector in the Monte Carlo
            simulator core.

            Fields
            ------
            direction: mc_point3f_t
                Reference direction/orientation of the detector.
            position: mc_point2f_t
                Position of the center/origin of the x-frequency detector.
            f_min: mc_fp_t
                The first spatial frequency point (rad/m).
            df: mc_fp_t
                Separation of the spatial frequency points (rad/m).
            cos_min: mc_fp_t
                Cosine of the maximum acceptance angle (relative to the
                direction of the detector).
            n: mc_size_t
                The number of spatial frequency points.
            offset: mc_size_t
                The offset of the first accumulator in the Monte Carlo
                detector buffer.

            Note
            ----
            The buffer holds 4*n accumulators.
            For frequency index f:
              - positive real  -> offset + 4*f
              - negative real  -> offset + 4*f + 1
              - positive imag  -> offset + 4*f + 2
              - negative imag  -> offset + 4*f + 3
            Net complex reflectance: (pos_re - neg_re) + 1j*(pos_im - neg_im).
            '''
            _fields_ = [
                ('direction', T.mc_point3f_t),
                ('position', T.mc_point2f_t),
                ('f_min', T.mc_fp_t),
                ('df', T.mc_fp_t),
                ('cos_min', T.mc_fp_t),
                ('n', T.mc_size_t),
                ('offset', T.mc_size_t),
            ]
        return ClXFrequency

    def cl_declaration(self, mc: mcobject.McObject) -> str:
        '''
        Structure that defines the detector in the Monte Carlo simulator.
        '''
        loc = self.location
        Loc = loc.capitalize()
        return '\n'.join((
            'struct MC_STRUCT_ATTRIBUTES Mc{}Detector{{'.format(Loc),
            '	mc_point3f_t direction;'
            '	mc_point2f_t position;'
            '	mc_fp_t f_min;',
            '	mc_fp_t df;',
            '	mc_fp_t cos_min;',
            '	mc_size_t n;',
            '	mc_size_t offset;',
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
            '	dbg_print("Mc{}Detector - XFrequency detector:");'.format(Loc),
            '	dbg_print_point3f(INDENT "direction:", &detector->direction);',
            '	dbg_print_point2f(INDENT "position:", &detector->position);',
            '	dbg_print_float(INDENT "f_min (1/mm):", detector->f_min*1e-3f);',
            '	dbg_print_float(INDENT "df (1/mm):", detector->df*1e-3f);',
            '	dbg_print_float(INDENT "cos_min:", detector->cos_min);',
            '	dbg_print_size_t(INDENT "n:", detector->n);',
            '	dbg_print_size_t(INDENT "offset:", detector->offset);',
            '};',
            '',
            'inline void mcsim_{}_detector_deposit('.format(loc),
            '		McSim *mcsim,',
            '		mc_point3f_t const *pos, mc_point3f_t const *dir,',
            '		mc_fp_t weight){',
            '',
            '	__global mc_accu_t *address;',
            '	mc_fp_t weight_sfd_real, weight_sfd_imag, frequency;',
            '	uint32_t ui32w;',
            '',
            '	dbg_print_status(mcsim, "{} XFrequency detector hit");'.format(Loc),
            '',
            '	__mc_detector_mem const struct Mc{}Detector *detector ='.format(Loc),
            '		mcsim_{}_detector(mcsim);'.format(loc),
            '',
            '	mc_fp_t dx = pos->x - detector->position.x;',
            '',
            '	mc_point3f_t detector_direction = detector->direction;',
            '	mc_int_t directional_factor = (detector->cos_min <= mc_fabs(mc_dot_point3f(dir, &detector_direction))) ? 1 : 0;',
            '',
            '	for (mc_int_t f_index = 0; f_index < detector->n; f_index++) {',
            '		frequency = detector->f_min + f_index*detector->df;',
            '		weight_sfd_real = weight * mc_cos(FP_2PI * frequency * dx);',
            '		weight_sfd_imag = weight * mc_sin(FP_2PI * frequency * dx);',
            '',
            '		/* Each frequency uses four accumulators:',
            '		   [4*f+0] pos real, [4*f+1] neg real, [4*f+2] pos imag, [4*f+3] neg imag */',
            '',
            '		/* real part */',
            '		if (weight_sfd_real >= FP_0) {',
            '			address = mcsim_accumulator_buffer_ex(mcsim, detector->offset + 4*f_index);',
            '			ui32w = weight_to_int(weight_sfd_real) * directional_factor;',
            '		} else {',
            '			address = mcsim_accumulator_buffer_ex(mcsim, detector->offset + 4*f_index + 1);',
            '			ui32w = weight_to_int(-weight_sfd_real) * directional_factor;',
            '		}',
            '		if (ui32w > 0) {',
            '			dbg_print_uint("{} XFrequency detector depositing real int:", ui32w);'.format(Loc),
            '			accumulator_deposit(address, ui32w);',
            '		}',
            '',
            '		/* imaginary part */',
            '		if (weight_sfd_imag >= FP_0) {',
            '			address = mcsim_accumulator_buffer_ex(mcsim, detector->offset + 4*f_index + 2);',
            '			ui32w = weight_to_int(weight_sfd_imag) * directional_factor;',
            '		} else {',
            '			address = mcsim_accumulator_buffer_ex(mcsim, detector->offset + 4*f_index + 3);',
            '			ui32w = weight_to_int(-weight_sfd_imag) * directional_factor;',
            '		}',
            '		if (ui32w > 0) {',
            '			dbg_print_uint("{} XFrequency detector depositing imag int:", ui32w);'.format(Loc),
            '			accumulator_deposit(address, ui32w);',
            '		}',
            '	}',
            '};'
        ))

    def __init__(self, faxis: axis.EdgeAxis,
                 position: Tuple[float, float] = (0.0, 0.0),
                 cosmin: float = 0.0,
                 direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)):
        '''
        X-direction spatial frequency accumulator.

        Accumulates the complex SFD reflectance along the x direction using
        cos/sin kernel weights (1D Fourier transform instead of Hankel).
        The raw buffer has shape ``(4*n,)``:
        ``[4*f+0]`` positive real, ``[4*f+1]`` negative real,
        ``[4*f+2]`` positive imag, ``[4*f+3]`` negative imag.
        Net complex reflectance:
        ``(raw[0::4] - raw[1::4]) + 1j*(raw[2::4] - raw[3::4])``.

        Parameters
        ----------
        faxis: axis.EdgeAxis
            Object that defines the spatial frequency points (rad/m).
        position: (float, float)
            Position of the center of the x-frequency accumulator (x, y) in meters.
        cosmin: float
            Cosine of the maximum acceptance angle (relative to the direction
            of the detector).
        direction: (float, float, float)
            Reference direction / orientation of the detector.
        '''
        if isinstance(faxis, XFrequency):
            xfrequency = faxis
            position = xfrequency.position
            faxis = type(xfrequency.faxis)(xfrequency.faxis)
            cosmin = xfrequency.cosmin
            direction = xfrequency.direction
            raw_data = np.copy(xfrequency.raw)
            nphotons = xfrequency.nphotons
        else:
            raw_data = np.zeros((4 * faxis.n,))
            nphotons = 0

        super().__init__(raw_data, nphotons)

        self._position = np.zeros((2,))
        self._cosmin = 0.0
        self._direction = np.zeros((3,))
        self._f_axis = faxis
        self._set_position(position)
        self._set_cosmin(cosmin)
        self._set_direction(direction)

    def _get_faxis(self) -> axis.EdgeAxis:
        return self._f_axis
    faxis = property(_get_faxis, None, None, 'Frequency axis object.')

    def _get_frequencies(self) -> np.ndarray:
        return self._f_axis.centers
    frequencies = property(_get_frequencies, None, None,
                           'Frequencies at which the reflectance is accumulated (rad/m).')

    def _get_position(self) -> Tuple[float, float]:
        return self._position
    def _set_position(self, value: float or Tuple[float, float]):
        self._position[:] = value
    position = property(_get_position, _set_position, None,
                        'Position of the x-frequency accumulator as a tuple (x, y).')

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
        self._direction *= 1.0 / norm
    direction = property(_get_direction, _set_direction, None,
                         'Detector reference direction.')

    def _get_n(self) -> int:
        return self._f_axis.n
    n = property(_get_n, None, None, 'Number of frequency accumulators.')

    def _get_normalized(self) -> np.ndarray:
        '''
        Net complex SFD reflectance as a (n,) array.
        raw has shape (4*n,):
          [4*f+0] positive real, [4*f+1] negative real,
          [4*f+2] positive imag, [4*f+3] negative imag.
        '''
        k = 1.0 / max(self._nphotons, 1)
        real_part = (self.raw[0::4] - self.raw[1::4]) * k
        imag_part = (self.raw[2::4] - self.raw[3::4]) * k
        return real_part + 1j * imag_part
    normalized = property(_get_normalized, None, None, 'Normalized.')
    reflectance = property(_get_normalized, None, None, 'Reflectance.')
    transmittance = property(_get_normalized, None, None, 'Transmittance.')

    def cl_pack(self, mc: mcobject.McObject,
                target: cltypes.Structure = None) -> cltypes.Structure:
        '''
        Fills the structure (target) with the data required by the
        Monte Carlo simulator.
        See the :py:meth:`XFrequency.cl_type` method for a detailed list of fields.

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
        target.n = self._f_axis.n

        target.cos_min = self._cosmin

        target.direction.fromarray(self._direction)

        return target

    def todict(self):
        '''
        Save the accumulator configuration without the accumulator data to
        a dictionary. Use the :meth:`XFrequency.fromdict` method to create a new
        accumulator instance from the returned data.

        Returns
        -------
        data: dict
            Accumulator configuration as a dictionary.
        '''
        return {
            'type': 'XFrequency',
            'position': self._position.tolist(),
            'f_axis': self._f_axis.todict(),
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
            Dictionary created by the :py:meth:`XFrequency.todict` method.
        '''
        data = dict(data)
        rt_type = data.pop('type')
        if rt_type != 'XFrequency':
            raise TypeError(
                'Expected "XFrequency" type but got "{}"!'.format(rt_type))
        f_axis_data = data.pop('f_axis')
        f_axis_type = f_axis_data.pop('type')

        return XFrequency(getattr(axis, f_axis_type)(**f_axis_data), **data)

    def __str__(self):
        return 'XFrequency(f_axis={}, position=({}, {}), cosmin={}, '\
               'direction=({}, {}, {}))'.format(
                   self._f_axis, *self._position, self._cosmin,
                   *self._direction)

    def __repr__(self):
        return '{} #{}'.format(self.__str__(), id(self))
