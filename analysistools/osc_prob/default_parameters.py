# -*- coding: utf-8 -*-
from numpy import radians, sqrt

theta_12_BF = radians(33.44)
theta_13_BF = radians(8.57)
theta_23_BF = radians(49.2)
delta_cp_BF = radians(197)
dm2_21_BF = 7.42e-5
dm2_32_BF = 2.517e-3
dm2_31_BF = dm2_21_BF + dm2_32_BF

baseline = 295.0

SQRT3 = sqrt(3.0)
r"""float: Module-level constant

Constant equal to sqrt(3.0).
"""

SQRT3_INV = 1./sqrt(3.0)
r"""float: Module-level constant

Constant equal to 1.0/sqrt(3.0).
"""

NEG_HALF_SQRT3_INV = -SQRT3_INV/2.0
r"""float: Module-level constant

Constant equal to -1.0/sqrt(3.0)/2.0.
"""



CONV_KM_TO_INV_EV = 5.06773e9
r"""float: Module-level constant

Multiplicative conversion factor from km to eV^{-1}.
Units: [km^{-1} eV^{-1}].
"""

CONV_CM_TO_INV_EV = CONV_KM_TO_INV_EV*1.e-5
r"""float: Module-level constant

Multiplicative conversion factor from cm to eV^{-1}.
Units: [cm^{-1} eV^{-1}]
"""

CONV_INV_EV_TO_CM = 1./CONV_CM_TO_INV_EV
r"""float: Module-level constant

Multiplicative conversion factor from eV^{-1} to cm.
Units: [eV cm]
"""

CONV_EV_TO_G = 1.783e-33
r"""float: Module-level constant

Multiplicative conversion factor from eV^{-1} to grams.
Units: [g eV^{-1}]
"""

CONV_G_TO_EV = 1./CONV_EV_TO_G
r"""float: Module-level constant

Multiplicative conversion factor from grams to eV^{-1}.
Units: [eV g^{-1}]
"""

GF = 1.1663787e-23
r"""float: Module-level constant

Fermi constant.
Units: [eV^{-1}]
"""

MASS_ELECTRON = 0.5109989461e6
r"""float: Module-level constant

Electron mass.
Units: [eV]
"""

MASS_PROTON = 938.272046e6
r"""float: Module-level constant

Proton mass.
Units: [eV]
"""

MASS_NEUTRON = 939.565379e6
r"""float: Module-level constant

Neutron mass.
Units: [eV]
"""

ELECTRON_FRACTION_EARTH_CRUST = 0.5
r"""float: Module-level constant

Electron fraction in the Earth's crust.
Units: [Adimensional]
"""

DENSITY_MATTER_CRUST_G_PER_CM3 = 3.0
r"""float: Module-level constant

Average matter density in the Earth's crust.
Units: [g cm^{-3}]
"""

NUM_DENSITY_E_EARTH_CRUST = DENSITY_MATTER_CRUST_G_PER_CM3 * CONV_G_TO_EV \
                            / ((MASS_PROTON+MASS_NEUTRON)/2.0) \
                            * ELECTRON_FRACTION_EARTH_CRUST \
                            / pow(CONV_CM_TO_INV_EV, 3.0)
r"""float: Module-level constant

Electron number density in the Earth's crust
Units: [eV^3]
"""

VCC_EARTH_CRUST = sqrt(2.0)*GF*NUM_DENSITY_E_EARTH_CRUST
r"""float: Module-level constant

Charged-current matter potential in the Earth's crust.
Units: [eV]
"""