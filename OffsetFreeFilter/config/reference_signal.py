# reference signal functions
#
#
# Requirements:
# * Python 3
#
# Copyright (c) 2022 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import sys
sys.dont_write_bytecode = True


def myRef(t, ts, ref=1.5):

    # if t >= 60 and t < 120:
    #     ref *= 1.1
    # elif t >= 120 and t < 180:
    #     ref *= 0.9
    # elif t >= 180 and t < 240:
    #     ref *= 1.15
    # elif t >= 240 and t < 300:
    #     ref *= 0.85

    return ref
