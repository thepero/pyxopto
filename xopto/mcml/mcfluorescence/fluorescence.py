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

from xopto.mcml import mcobject, cltypes


def alias_table(pdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Build a Walker/Vose alias table for sampling a discrete distribution
    in O(1). A sample is obtained from a single uniform random number u
    from [0, 1) as x = u*n, i = floor(x), and returning i if
    x - i < prob[i] else alias[i].

    Parameters
    ----------
    pdf: np.ndarray
        Discrete probabilities (need not be normalized).

    Returns
    -------
    prob: np.ndarray
        Acceptance probabilities of the bins.
    alias: np.ndarray
        Alias bin indices.
    '''
    pdf = np.asarray(pdf, dtype=np.float64)
    n = pdf.size
    scaled = pdf*(n/pdf.sum())
    prob = np.ones(n)
    alias = np.arange(n)
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)
    # numerical leftovers are accepted with certainty
    for i in large + small:
        prob[i] = 1.0
    return prob, alias


def apply_quantum_yield(data: np.ndarray, qy: float, axis: int = 0) -> np.ndarray:
    '''
    Apply the fluorescence quantum yield to generation-resolved detector
    data that were simulated with a quantum yield of 1.

    A photon packet that underwent m emission events survives with
    probability qy**m, hence the result is sum_m qy**m*data[m].
    The last generation bin collects all packets with m >= M and is
    weighted with qy**M.

    Parameters
    ----------
    data: np.ndarray
        Generation-resolved data (generation axis given by axis).
    qy: float
        Fluorescence quantum yield.
    axis: int
        Generation axis of the data.

    Returns
    -------
    result: np.ndarray
        Data with the generation axis removed.
    '''
    data = np.asarray(data)
    shape = [1]*data.ndim
    shape[axis] = data.shape[axis]
    weights = float(qy)**np.arange(data.shape[axis])
    return np.sum(data*weights.reshape(shape), axis=axis)


def photon_to_energy(data: np.ndarray, wavelengths: np.ndarray,
                     excitation: float, axis: int = 0) -> np.ndarray:
    '''
    Convert photon-number weights to energy weights relative to the energy
    of the excitation photons, i.e. multiply by excitation/wavelength.

    Parameters
    ----------
    data: np.ndarray
        Spectrally resolved data.
    wavelengths: np.ndarray
        Wavelengths of the spectral bins (m).
    excitation: float
        Excitation wavelength (m).
    axis: int
        Wavelength axis of the data.

    Returns
    -------
    energy: np.ndarray
        Energy weighted data.
    '''
    data = np.asarray(data)
    shape = [1]*data.ndim
    shape[axis] = data.shape[axis]
    scale = float(excitation)/np.asarray(wavelengths, dtype=np.float64)
    return data*scale.reshape(shape)


class SpectralDetection:
    def __init__(self, start: float = None, stop: float = None,
                 binsize: int = 1, max_generation: int = 4):
        '''
        Spectral and generation resolution of a detector used in
        fluorescence simulations. The accumulated data of the detector
        get two leading axes (generation, wavelength).

        Parameters
        ----------
        start: float
            First detected wavelength (m). Defaults to the first wavelength of
            the simulation grid. Rounded to the nearest grid point.
        stop: float
            Last detected wavelength (m). Defaults to the last wavelength of
            the simulation grid. Rounded to the nearest grid point.
        binsize: int
            Number of simulation grid wavelengths that are accumulated in one
            wavelength bin of the detector.
        max_generation: int
            Photon packets are binned by the number of fluorescence emission
            events m = 0, 1, ..., max_generation. The last bin collects all
            photon packets with m >= max_generation.
        '''
        self.start = start
        self.stop = stop
        self.binsize = max(int(binsize), 1)
        self.max_generation = max(int(max_generation), 0)

    def resolve(self, fluorescence: 'Fluorescence') -> Tuple[int, int, int, int]:
        '''
        Resolve the detection range on the simulation wavelength grid.

        Returns
        -------
        first: int
            Index of the first detected grid wavelength.
        binsize: int
            Number of grid wavelengths per bin.
        n: int
            Number of wavelength bins.
        max_generation: int
            Index of the last generation bin.
        '''
        wl = fluorescence.wavelengths
        first = 0 if self.start is None else fluorescence.index(self.start)
        last = wl.size - 1 if self.stop is None else fluorescence.index(self.stop)
        if last < first:
            raise ValueError('The last detected wavelength must not be '
                             'shorter than the first detected wavelength!')
        n = (last - first)//self.binsize + 1
        return first, self.binsize, n, self.max_generation

    def wavelengths(self, fluorescence: 'Fluorescence') -> np.ndarray:
        ''' Center wavelengths of the detector wavelength bins (m). '''
        first, binsize, n, _ = self.resolve(fluorescence)
        wl = fluorescence.wavelengths
        return np.array([wl[first + i*binsize:
                            min(first + (i + 1)*binsize, wl.size)].mean()
                         for i in range(n)])

    def todict(self) -> dict:
        return {'start': self.start, 'stop': self.stop, 'binsize': self.binsize,
                'max_generation': self.max_generation,
                'type': 'SpectralDetection'}

    def __str__(self):
        return 'SpectralDetection(start={}, stop={}, binsize={}, '\
               'max_generation={})'.format(self.start, self.stop,
                                           self.binsize, self.max_generation)

    def __repr__(self):
        return '{:s} # id 0x{:>08X}.'.format(self.__str__(), id(self))


def spectral_detection_cl_type(mc: mcobject.McObject) -> cltypes.Structure:
    '''
    OpenCL structure type of the spectral detection parameters
    (McDetectorSpectral) that is included in the spectral detectors.
    '''
    T = mc.types
    class ClDetectorSpectral(cltypes.Structure):
        _fields_ = [
            ('first', T.mc_int_t),
            ('binsize', T.mc_int_t),
            ('n', T.mc_int_t),
            ('max_generation', T.mc_int_t),
            ('stride', T.mc_size_t),
        ]
    return ClDetectorSpectral


class Fluorescence(mcobject.McObject):
    '''
    Configuration of a fluorescence cascade simulation. Enables the
    spectral layers (:py:class:`~xopto.mcml.mclayer.layer.SpectralLayer`),
    the re-emission of photon packets that are absorbed by fluorophores and
    the spectrally and generation resolved detectors.
    '''

    def cl_options(self, mc: mcobject.McObject):
        return [('MC_USE_FLUORESCENCE', True),
                ('MC_FLUORESCENCE_QY_IN_KERNEL', self._qy_in_kernel),
                ('MC_USE_FP_LUT', True),
                ('MC_METHOD', 1)]  # albedo rejection

    def cl_type(self, mc: mcobject.McObject) -> cltypes.Structure:
        T = mc.types
        class ClFluorescence(cltypes.Structure):
            '''
            Fields
            ------
            num_wavelengths: mc_int_t
                Number of wavelengths of the simulation grid.
            num_fluorophores: mc_int_t
                Number of fluorophore slots per layer.
            excitation_index: mc_int_t
                Wavelength index of the launched photon packets.
            absorbers_offset: mc_size_t
                Offset of the cumulative fluorophore absorption probabilities
                [layer][wavelength][fluorophore] in the lookup table buffer.
            qy_offset: mc_size_t
                Offset of the quantum yields [layer][fluorophore].
            alias_offset: mc_size_t
                Offset of the emission alias tables
                [layer][fluorophore][wavelength][probability, alias].
            '''
            _fields_ = [
                ('num_wavelengths', T.mc_int_t),
                ('num_fluorophores', T.mc_int_t),
                ('excitation_index', T.mc_int_t),
                ('absorbers_offset', T.mc_size_t),
                ('qy_offset', T.mc_size_t),
                ('alias_offset', T.mc_size_t),
            ]
        return ClFluorescence

    def cl_declaration(self, mc: mcobject.McObject) -> str:
        return '\n'.join((
            '/**',
            ' * @brief Spectral and generation resolution of a detector.',
            ' */',
            'struct MC_STRUCT_ATTRIBUTES McDetectorSpectral{',
            '	mc_int_t first;          /**< Index of the first detected wavelength. */',
            '	mc_int_t binsize;        /**< Number of grid wavelengths in one bin. */',
            '	mc_int_t n;              /**< Number of wavelength bins. */',
            '	mc_int_t max_generation; /**< Last generation bin (collects >= max_generation). */',
            '	mc_size_t stride;        /**< Number of accumulators of one wavelength bin. */',
            '};',
            'typedef struct McDetectorSpectral McDetectorSpectral;',
            '',
            '/**',
            ' * @brief Fluorescence configuration.',
            ' */',
            'struct MC_STRUCT_ATTRIBUTES McFluorescence{',
            '	mc_int_t num_wavelengths;   /**< Number of wavelengths. */',
            '	mc_int_t num_fluorophores;  /**< Number of fluorophore slots per layer. */',
            '	mc_int_t excitation_index;  /**< Wavelength index of launched packets. */',
            '	mc_size_t absorbers_offset; /**< Cumulative absorption probabilities [layer][wl][fluorophore]. */',
            '	mc_size_t qy_offset;        /**< Quantum yields [layer][fluorophore]. */',
            '	mc_size_t alias_offset;     /**< Emission alias tables [layer][fluorophore][wl][2]. */',
            '};',
            'typedef struct McFluorescence McFluorescence;',
        ))

    def cl_implementation(self, mc: mcobject.McObject) -> str:
        return '\n'.join((
            '#if MC_ENABLE_DEBUG',
            'void dbg_print_fluorescence(__mc_fluorescence_mem const McFluorescence *fl){',
            '	dbg_print("McFluorescence:");',
            '	dbg_print_int(INDENT "num_wavelengths:", fl->num_wavelengths);',
            '	dbg_print_int(INDENT "num_fluorophores:", fl->num_fluorophores);',
            '	dbg_print_int(INDENT "excitation_index:", fl->excitation_index);',
            '};',
            '#endif',
            '',
            '/**',
            ' * @brief Handles an absorption event. Selects the absorber and, if',
            ' *        a fluorophore absorbed the photon packet, re-emits the packet',
            ' *        in an isotropic direction at a wavelength sampled from the',
            ' *        emission spectrum of the fluorophore.',
            ' * @param[in, out] psim Simulator instance.',
            ' * @return Nonzero if the packet was re-emitted, zero if it was',
            ' *         absorbed (by the base medium or a non-radiative decay).',
            ' */',
            'inline int mcsim_fluorescence_absorb(McSim *psim){',
            '	__mc_fluorescence_mem const McFluorescence *fl = mcsim_fluorescence(psim);',
            '	mc_int_t num_fl = fl->num_fluorophores;',
            '	mc_int_t num_wl = fl->num_wavelengths;',
            '	mc_int_t layer = mcsim_current_layer_index(psim);',
            '',
            '	/* select the absorber from the cumulative probabilities */',
            '	__mc_fp_lut_mem mc_fp_t const *cumulative = mcsim_fp_lut_array_ex(',
            '		psim, fl->absorbers_offset +',
            '		(layer*num_wl + mcsim_wavelength_index(psim))*num_fl);',
            '	mc_fp_t u = mcsim_random(psim);',
            '	mc_int_t k = 0;',
            '	while (k < num_fl && u >= cumulative[k])',
            '		++k;',
            '	if (k >= num_fl)',
            '		return 0; /* absorbed by the base medium */',
            '',
            '	#if MC_FLUORESCENCE_QY_IN_KERNEL',
            '	if (mcsim_random(psim) >= ',
            '			mcsim_fp_lut_array_ex(psim, fl->qy_offset)[layer*num_fl + k])',
            '		return 0; /* non-radiative decay */',
            '	#endif',
            '',
            '	/* sample the emission wavelength from the alias table */',
            '	__mc_fp_lut_mem mc_fp_t const *alias = mcsim_fp_lut_array_ex(',
            '		psim, fl->alias_offset + 2*(layer*num_fl + k)*num_wl);',
            '	mc_fp_t x = mcsim_random(psim)*num_wl;',
            '	mc_int_t i = (mc_int_t)x;',
            '	i = (i < num_wl) ? i : num_wl - 1;',
            '	mc_int_t wl_index = (x - i < alias[2*i]) ? i : (mc_int_t)alias[2*i + 1];',
            '	mcsim_set_wavelength_index(psim, wl_index);',
            '	psim->state.generation += 1;',
            '',
            '	/* isotropic emission */',
            '	mc_fp_t sin_fi, cos_fi;',
            '	mc_fp_t cos_theta = FP_1 - FP_2*mcsim_random(psim);',
            '	mc_fp_t sin_theta = mc_sqrt(mc_fmax(FP_1 - cos_theta*cos_theta, FP_0));',
            '	mc_sincos(FP_2PI*mcsim_random(psim), &sin_fi, &cos_fi);',
            '	mc_point3f_t dir = {cos_fi*sin_theta, sin_fi*sin_theta, cos_theta};',
            '	mcsim_set_direction(psim, &dir);',
            '',
            '	return 1;',
            '};',
            '',
            '/**',
            ' * @brief Computes the accumulator offset of the current photon packet',
            ' *        wavelength and generation.',
            ' * @param[in] psim Simulator instance.',
            ' * @param[in] spectral Spectral detection parameters of the detector.',
            ' * @param[out] offset Accumulator offset.',
            ' * @return Nonzero if the wavelength is within the detected range.',
            ' */',
            'inline int mcsim_detector_spectral_offset(',
            '		McSim const *psim,',
            '		__mc_detector_mem McDetectorSpectral const *spectral,',
            '		mc_size_t *offset){',
            '	mc_int_t bin = mcsim_wavelength_index(psim) - spectral->first;',
            '	if (bin < 0)',
            '		return 0;',
            '	bin /= spectral->binsize;',
            '	if (bin >= spectral->n)',
            '		return 0;',
            '	mc_int_t generation = mcsim_generation(psim);',
            '	if (generation > spectral->max_generation)',
            '		generation = spectral->max_generation;',
            '	*offset = ((mc_size_t)generation*spectral->n + bin)*spectral->stride;',
            '	return 1;',
            '};',
        ))

    def __init__(self, wavelengths: np.ndarray, excitation: float,
                 qy_in_kernel: bool = False):
        '''
        Fluorescence cascade configuration.

        Parameters
        ----------
        wavelengths: np.ndarray
            Uniform wavelength grid (m). All the spectra of the spectral
            layers and fluorophores are defined on this grid.
        excitation: float
            Wavelength of the launched photon packets (m). Must be a point
            of the wavelength grid.
        qy_in_kernel: bool
            If True, the quantum yield of the fluorophores is applied in the
            kernel (photon packets that fail the quantum yield test are
            terminated). If False (default), all photon packets absorbed by
            a fluorophore are re-emitted and the quantum yield is applied to
            the generation-resolved detector data after the simulation
            (see :py:func:`apply_quantum_yield`). This requires that all
            the fluorophores share the same quantum yield.
        '''
        wavelengths = np.asarray(wavelengths, dtype=np.float64).ravel()
        if wavelengths.size < 1:
            raise ValueError('The wavelength grid must not be empty!')
        if wavelengths.size > 1:
            steps = np.diff(wavelengths)
            if np.any(steps <= 0.0) or \
                    not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
                raise ValueError('The wavelength grid must be uniform and '
                                 'increasing!')
        self._wavelengths = wavelengths
        self._excitation_index = self.index(excitation)
        self._qy_in_kernel = bool(qy_in_kernel)

    def index(self, wavelength: float) -> int:
        '''
        Index of the grid point that matches the given wavelength.
        Raises ValueError if the wavelength is not a grid point.
        '''
        wl = self._wavelengths
        index = int(np.argmin(np.abs(wl - float(wavelength))))
        tol = 1e-3*(wl[1] - wl[0]) if wl.size > 1 else 1e-12
        if abs(wl[index] - float(wavelength)) > tol:
            raise ValueError('Wavelength {} m is not a point of the '
                             'wavelength grid!'.format(wavelength))
        return index

    def _get_wavelengths(self) -> np.ndarray:
        return self._wavelengths
    wavelengths = property(_get_wavelengths, None, None,
                           'Wavelength grid (m).')

    def _get_num_wavelengths(self) -> int:
        return self._wavelengths.size
    num_wavelengths = property(_get_num_wavelengths, None, None,
                               'Number of wavelengths of the grid.')

    def _get_excitation(self) -> float:
        return float(self._wavelengths[self._excitation_index])
    def _set_excitation(self, wavelength: float):
        self._excitation_index = self.index(wavelength)
    excitation = property(_get_excitation, _set_excitation, None,
                          'Excitation wavelength (m).')

    def _get_excitation_index(self) -> int:
        return self._excitation_index
    excitation_index = property(_get_excitation_index, None, None,
                                'Wavelength index of the excitation.')

    def _get_qy_in_kernel(self) -> bool:
        return self._qy_in_kernel
    qy_in_kernel = property(_get_qy_in_kernel, None, None,
                            'Quantum yield applied in the kernel.')

    def cl_pack(self, mc: mcobject.McObject,
                target: cltypes.Structure = None) -> cltypes.Structure:
        '''
        Pack the fluorescence configuration and the lookup tables
        (absorber probabilities, quantum yields, emission alias tables)
        of all the layers.
        '''
        if target is None:
            target = self.fetch_cl_type(mc)()

        layers = list(mc.layers)
        num_layers = len(layers)
        num_wl = self.num_wavelengths
        fluorophores = [list(getattr(layer, 'fluorophores', ()))
                        for layer in layers]
        num_fl = max(1, max(len(item) for item in fluorophores))

        absorbers = np.zeros((num_layers, num_wl, num_fl))
        qy = np.zeros((num_layers, num_fl))
        alias = np.zeros((num_layers, num_fl, num_wl, 2))
        alias[..., 0] = 1.0
        alias[..., 1] = np.arange(num_wl)

        qys = set()
        for index, (layer, items) in enumerate(zip(layers, fluorophores)):
            for k, fluorophore in enumerate(items):
                fluorophore.check(num_wl)
                qy[index, k] = fluorophore.qy
                qys.add(fluorophore.qy)
                prob, table = alias_table(fluorophore.emission)
                alias[index, k, :, 0] = prob
                alias[index, k, :, 1] = table
            if items:
                for w in range(num_wl):
                    mua_total = layer.mua_at(w)
                    if mua_total > 0.0:
                        mua = np.array([item.mua_at(w) for item in items])
                        absorbers[index, w, :len(items)] = \
                            np.cumsum(mua)/mua_total

        if not self._qy_in_kernel and len(qys) > 1:
            raise ValueError(
                'The quantum yield can be applied after the simulation only '
                'if all the fluorophores share the same quantum yield! '
                'Use qy_in_kernel=True.')

        np_float = mc.types.np_float
        target.num_wavelengths = num_wl
        target.num_fluorophores = num_fl
        target.excitation_index = self._excitation_index
        target.absorbers_offset = mc.append_r_lut(
            np.asarray(absorbers, dtype=np_float).ravel()).offset
        target.qy_offset = mc.append_r_lut(
            np.asarray(qy, dtype=np_float).ravel()).offset
        target.alias_offset = mc.append_r_lut(
            np.asarray(alias, dtype=np_float).ravel()).offset

        return target

    def todict(self) -> dict:
        return {'wavelengths': self._wavelengths.tolist(),
                'excitation': self.excitation,
                'qy_in_kernel': self._qy_in_kernel,
                'type': 'Fluorescence'}

    @classmethod
    def fromdict(cls, data: dict) -> 'Fluorescence':
        data_ = dict(data)
        if data_.pop('type') != 'Fluorescence':
            raise ValueError('Cannot create a Fluorescence from the data!')
        return cls(**data_)

    def __str__(self):
        return 'Fluorescence(wavelengths=[{:g} ... {:g}] ({} points), '\
               'excitation={:g}, qy_in_kernel={})'.format(
                   self._wavelengths[0], self._wavelengths[-1],
                   self._wavelengths.size, self.excitation, self._qy_in_kernel)

    def __repr__(self):
        return '{:s} # id 0x{:>08X}.'.format(self.__str__(), id(self))
