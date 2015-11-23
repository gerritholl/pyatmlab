"""Relevant definitions for TOVS
"""

import numpy

# I don't know a practical way of extracting those automatically, as the
# NWPSAF document only lists them in comments/text, and I don't know how
# to automatically map the NWPsaf document to the KLM User's Guide.
# Values are obtained from NOAA KLM User's Guide, April 2014 revision.
#
# Note that scale factors are defined in powers of 10, so a scale factor
# of 1 still means a multiplication factor of 10.
#
# Scale factor should either be scalar, or match the size of one line of
# data.

# HIRS/3, headers: Table 8.3.1.5.2.1-1., page 8-98 – 8-115
# HIRS/3, data: Table 8.3.1.5.3.1-1., page 8-142 — 8-169
# HIRS/4, headers: Table 8.3.1.5.2.2-1., page 8-115 – 8-142
# HIRS/4, data: Table 8.3.1.5.3.2-1., page 8-169 – 8-187

HIRS_scale_factors = {3: dict(
    hrs_h_calinf = 6,
    hrs_h_tempradcnv = numpy.concatenate((numpy.tile(6, 12*3), numpy.tile((5, 6, 6), 7))),
    hrs_h_20solfiltirrad = 6,
    hrs_h_20equifiltwidth = 6,
    hrs_h_nadloctol = 1,
    hrs_h_rollerr = 3,
    hrs_h_pitcherr = 3,
    hrs_h_yawerr = 3,
    hrs_h_smaxis = 5,
    hrs_h_eccen = 8,
    hrs_h_incli = 5,
    hrs_h_argper = 5,
    hrs_h_rascnod = 5,
    hrs_h_manom = 5,
    hrs_h_xpos = 5,
    hrs_h_ypos = 5,
    hrs_h_zpos = 5,
    hrs_h_xvel = 8,
    hrs_h_yvel = 8,
    hrs_h_zvel = 8,
    hrs_h_earthsun = 6,
    hrs_h_rdtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_bptemp = (2, 2, 3, 3, 3, 5),
    hrs_h_eltemp = (2, 2, 3, 3, 3, 5),
    hrs_h_pchtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_fhcc = (2, 2, 3, 3, 3, 5),
    hrs_h_scnmtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_fwmtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_p5v = (2, 2, 3, 3, 3, 5),
    hrs_h_p10v = (2, 2, 3, 3, 3, 5),
    hrs_h_p75v = (2, 2, 3, 3, 3, 5),
    hrs_h_m75v = (2, 2, 3, 3, 3, 5),
    hrs_h_p15v = (2, 2, 3, 3, 3, 5),
    hrs_h_m15v = (2, 2, 3, 3, 3, 5),
    hrs_h_fwmcur = (2, 2, 3, 3, 3, 5),
    # NOTE: In KLM User's Guide on page 8-114, Scan Motor Coefficient 2 is
    # missing.  Assuming this to be the same as coefficient 2 for all the
    # surrounding headers.
    hrs_h_scmcur = (2, 2, 3, 3, 3, 5),
    hrs_h_pchcpow = (2, 2, 3, 3, 3, 5),
    hrs_calcof = numpy.tile((12, 9, 6), 20),
    hrs_scalcof = numpy.tile((12, 9, 6), 20),
    hrs_rollang = 3,
    hrs_pitchang = 3,
    hrs_yawang = 3,
    hrs_scalti = 1,
    hrs_ang = 2,
    hrs_pos = 4)}

HIRS_scale_factors[4] = HIRS_scale_factors[3].copy()
    
