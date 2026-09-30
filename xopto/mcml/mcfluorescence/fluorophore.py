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

import numpy as np


def _spectrum(value, name: str) -> np.ndarray:
    ''' Convert a scalar or an array-like spectrum to a 1D float array. '''
    data = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if data.ndim != 1:
        raise ValueError('The {} spectrum must be a scalar or a 1D '
                         'array!'.format(name))
    return data


class Fluorophore:
    def __init__(self, mua, emission, qy: float = 1.0, mus=0.0):
        '''
        A fluorescent component of a spectral layer.

        All the spectra are defined on the wavelength grid of the
        :py:class:`~xopto.mcml.mcfluorescence.fluorescence.Fluorescence`
        object that is passed to the simulator.

        Parameters
        ----------
        mua: float or np.ndarray
            Absorption (excitation) coefficient of the fluorophore (1/m) on
            the wavelength grid. Includes the concentration of the
            fluorophore, e.g. ln(10)*epsilon(lambda)*c.
        emission: np.ndarray
            Intrinsic emission spectrum on the wavelength grid. The spectrum
            is normalized internally to a discrete probability distribution
            over the grid points.
        qy: float
            Fluorescence quantum yield from [0, 1].
        mus: float or np.ndarray
            Scattering coefficient of the fluorophore (1/m). Added to the
            scattering coefficient of the layer. The fluorophore scatters
            with the scattering phase function of the layer.
        '''
        self._mua = _spectrum(mua, 'absorption')
        self._mus = _spectrum(mus, 'scattering')
        emission = _spectrum(emission, 'emission')
        if np.any(emission < 0.0) or not np.any(emission > 0.0):
            raise ValueError('The emission spectrum must be nonnegative '
                             'with at least one nonzero value!')
        self._emission = emission/emission.sum()
        self._qy = float(qy)
        if not 0.0 <= self._qy <= 1.0:
            raise ValueError('The quantum yield must be from [0, 1]!')

    def _get_mua(self) -> np.ndarray:
        return self._mua
    mua = property(_get_mua, None, None,
                   'Absorption coefficient spectrum (1/m).')

    def _get_mus(self) -> np.ndarray:
        return self._mus
    mus = property(_get_mus, None, None,
                   'Scattering coefficient spectrum (1/m).')

    def _get_emission(self) -> np.ndarray:
        return self._emission
    emission = property(_get_emission, None, None,
                        'Normalized emission spectrum (discrete probabilities).')

    def _get_qy(self) -> float:
        return self._qy
    qy = property(_get_qy, None, None, 'Fluorescence quantum yield.')

    def mua_at(self, index: int) -> float:
        ''' Absorption coefficient at the given wavelength index. '''
        return float(self._mua[index if self._mua.size > 1 else 0])

    def mus_at(self, index: int) -> float:
        ''' Scattering coefficient at the given wavelength index. '''
        return float(self._mus[index if self._mus.size > 1 else 0])

    def check(self, num_wavelengths: int):
        '''
        Check if the spectra are compatible with the wavelength grid.
        Raises ValueError on error.
        '''
        for name, data in (('absorption', self._mua), ('scattering', self._mus)):
            if data.size not in (1, num_wavelengths):
                raise ValueError(
                    'The {} spectrum of the fluorophore has {} values but the '
                    'wavelength grid has {} points!'.format(
                        name, data.size, num_wavelengths))
        if self._emission.size != num_wavelengths:
            raise ValueError(
                'The emission spectrum of the fluorophore has {} values but the '
                'wavelength grid has {} points!'.format(
                    self._emission.size, num_wavelengths))

    def todict(self) -> dict:
        ''' Export object to a dict. '''
        return {'mua': self._mua.tolist(), 'mus': self._mus.tolist(),
                'emission': self._emission.tolist(), 'qy': self._qy,
                'type': 'Fluorophore'}

    @classmethod
    def fromdict(cls, data: dict) -> 'Fluorophore':
        ''' Create a new object from a dict created by :py:meth:`todict`. '''
        data_ = dict(data)
        if data_.pop('type') != 'Fluorophore':
            raise ValueError('Cannot create a Fluorophore from the data!')
        return cls(**data_)

    def __str__(self):
        return 'Fluorophore(qy={}, mua=<{} values>, emission=<{} values>)'.format(
            self._qy, self._mua.size, self._emission.size)

    def __repr__(self):
        return '{:s} # id 0x{:>08X}.'.format(self.__str__(), id(self))
