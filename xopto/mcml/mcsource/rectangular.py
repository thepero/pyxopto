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

# Deprecated module, use xopto.mcml.mcsource.rectangularemitter instead.
# Will be removed in the next version.

import warnings

from xopto.mcml.mcsource.rectangularemitter import \
    UniformRectangularEmitter, LambertianRectangularEmitter, \
    UniformRectangularEmitterLut


def _warn(old: str, new: str):
    warnings.warn(
        '{} is deprecated and will be removed in the next version, '
        'use {} instead!'.format(old, new),
        FutureWarning, stacklevel=3)


class UniformRectangular(UniformRectangularEmitter):
    def __init__(self, *args, **kwargs):
        '''
        Deprecated, use :py:class:`UniformRectangularEmitter` instead.
        Will be removed in the next version.
        '''
        _warn('UniformRectangular', 'UniformRectangularEmitter')
        super().__init__(*args, **kwargs)


class LambertianRectangular(LambertianRectangularEmitter):
    def __init__(self, *args, **kwargs):
        '''
        Deprecated, use :py:class:`LambertianRectangularEmitter` instead.
        Will be removed in the next version.
        '''
        _warn('LambertianRectangular', 'LambertianRectangularEmitter')
        super().__init__(*args, **kwargs)


class UniformRectangularLut(UniformRectangularEmitterLut):
    def __init__(self, *args, **kwargs):
        '''
        Deprecated, use :py:class:`UniformRectangularEmitterLut` instead.
        Will be removed in the next version.
        '''
        _warn('UniformRectangularLut', 'UniformRectangularEmitterLut')
        super().__init__(*args, **kwargs)
