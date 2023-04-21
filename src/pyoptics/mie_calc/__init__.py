"""References:
    1. Novotny, L., & Hecht, B. (2012). Principles of Nano-Optics (2nd ed.).
       Cambridge: Cambridge University Press. doi:10.1017/CBO9780511794193
    2. Craig F. Bohren & Donald R. Huffman (1983). Absorption and Scattering of 
       Light by Small Particles. WILEY‐VCH Verlag GmbH & Co. KGaA. 
       doi:10.1002/9783527618156
"""
from .bead import Bead
from .interface import (
   fields_focus, 
   fields_focus_gaussian,
   fields_plane_wave,
   forces_focus,
)
from .objective import Objective
