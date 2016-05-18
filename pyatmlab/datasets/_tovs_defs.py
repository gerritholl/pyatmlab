"""Relevant definitions for TOVS
"""

import numpy
import collections

# Sources:
#
# For HIRS/2:
#
# - NOAA Polar Orbiter Data User's Guide, POD Guide,
# http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/podug/index.htm
# http://www1.ncdc.noaa.gov/pub/data/satellite/publications/podguides/TIROS-N%20thru%20N-14/pdf/
#   Chapter 2, Chapter 4, ...? 
#
# For HIRS/3 and HIRS/4:
#
# - NOAA KLM User's Guide,
# http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/klm/index.htm
# http://www1.ncdc.noaa.gov/pub/data/satellite/publications/podguides/N-15%20thru%20N-19/pdf/0.0%20NOAA%20KLM%20Users%20Guide.pdf
# -   and NWFSAF guide
# https://nwpsaf.eu/deliverables/aapp/NWPSAF-MF-UD-003_Formats.pdf
#
# HIRS/3, headers: Table 8.3.1.5.2.1-1., page 8-98 – 8-115
# HIRS/3, data: Table 8.3.1.5.3.1-1., page 8-142 — 8-169
# HIRS/4, headers: Table 8.3.1.5.2.2-1., page 8-115 – 8-142
# HIRS/4, data: Table 8.3.1.5.3.2-1., page 8-169 – 8-187

# I don't know a practical way of extracting scale factors automatically, as the
# NWPSAF document only lists them in comments/text, and I don't know how
# to automatically map the NWPsaf document to the KLM User's Guide.
# Values are obtained from NOAA KLM User's Guide, April 2014 revision.
#
# Note that scale factors are defined in powers of 10, so a scale factor
# of 1 still means a multiplication factor of 10.
#
# Scale factor should either be scalar, or match the size of one line of
# data.

HIRS_scale_factors = {}

_tmpsclfct = (6., 9., 14., 17., 21., 25.)
_tmpsclfct4 = numpy.tile(_tmpsclfct, 4)
_tmpsclfct5 = numpy.tile(_tmpsclfct, 5)

HIRS_scale_factors[3] = dict(
    hrs_h_calinf = 6,
    hrs_h_tempradcnv = numpy.concatenate((numpy.tile(6, 12*3), numpy.tile((5, 6, 6), 7))),
    hrs_h_iwtcnttmp = _tmpsclfct5,
    hrs_h_ictcnttmp = _tmpsclfct4,
    hrs_h_tttcnttmp = _tmpsclfct,
    hrs_h_fwcnttmp = _tmpsclfct4,
    hrs_h_patchexpcnttmp = _tmpsclfct,
    hrs_h_fsradcnttmp = _tmpsclfct,
    hrs_h_scmircnttmp = _tmpsclfct,
    hrs_h_pttcnttmp = _tmpsclfct,
    hrs_h_sttcnttmp = _tmpsclfct,
    hrs_h_bpcnttmp = _tmpsclfct,
    hrs_h_electcnttmp = _tmpsclfct,
    hrs_h_patchfcnttmp = _tmpsclfct,
    hrs_h_scmotcnttmp = _tmpsclfct,
    hrs_h_fwmcnttmp = _tmpsclfct,
    hrs_h_chsgcnttmp = _tmpsclfct,
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
    hrs_pos = 4)

HIRS_scale_factors[4] = HIRS_scale_factors[3].copy()

HIRS_scale_factors[2] = dict(
    # NB: normalisation coefficients have reversed order

    hrs_calcof = numpy.concatenate(
        (numpy.tile(numpy.array([44, 30, 22]), 40),
         numpy.tile(numpy.array([22, 30, 44]), 20)))
    )

HIRS_scale_bases = {}

HIRS_scale_bases[2] = 2
HIRS_scale_bases[3] = 10
HIRS_scale_bases[4] = 10

HIRS_header_dtypes = {}
HIRS_line_dtypes = {}

# Obtained using get_definition_from_PDF.  Please note correction!
HIRS_header_dtypes[3] = numpy.dtype([('hrs_h_siteid', '|S3', 1),
      ('hrs_h_blank', '|S1', 1),
      ('hrs_h_l1bversnb', '>i2', 1),
      ('hrs_h_l1bversyr', '>i2', 1),
      ('hrs_h_l1bversdy', '>i2', 1),
      ('hrs_h_reclg', '>i2', 1),
      ('hrs_h_blksz', '>i2', 1),
      ('hrs_h_hdrcnt', '>i2', 1),
      ('hrs_h_filler0', '>i2', 3),
      ('hrs_h_dataname', '|S42', 1),
      ('hrs_h_prblkid', '|S8', 1),
      ('hrs_h_satid', '>i2', 1),
      ('hrs_h_instid', '>i2', 1),
      ('hrs_h_datatyp', '>i2', 1),
      ('hrs_h_tipsrc', '>i2', 1),
      ('hrs_h_startdatajd', '>i4', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_startdatady', '>i2', 1),
      ('hrs_h_startdatatime', '>i4', 1),
      ('hrs_h_enddatajd', '>i4', 1),
      ('hrs_h_enddatayr', '>i2', 1),
      ('hrs_h_enddatady', '>i2', 1),
      ('hrs_h_enddatatime', '>i4', 1),
      ('hrs_h_cpidsyr', '>i2', 1),
      ('hrs_h_cpidsdy', '>i2', 1),
      ('hrs_h_filler1', '>i2', 4),
      ('hrs_h_inststat1', '>i4', 1),
      ('hrs_h_filler2', '>i2', 1),
      ('hrs_h_statchrecnb', '>i2', 1),
      ('hrs_h_inststat2', '>i4', 1),
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_callocsclin', '>i2', 1),
      ('hrs_h_misscnlin', '>i2', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_okdatafr', '>i2', 1),
      ('hrs_h_pacsparityerr', '>i2', 1),
      ('hrs_h_auxsyncerrsum', '>i2', 1),
      ('hrs_h_timeseqerr', '>i2', 1),
      ('hrs_h_timeseqerrcode', '>i2', 1),
      ('hrs_h_socclockupind', '>i2', 1),
      ('hrs_h_locerrind', '>i2', 1),
      ('hrs_h_locerrcode', '>i2', 1),
      ('hrs_h_pacsstatfield', '>i2', 1),
      ('hrs_h_pacsdatasrc', '>i2', 1),
      ('hrs_h_filler3', '>i4', 1),
      ('hrs_h_spare1', '|S8', 1),
      ('hrs_h_spare2', '|S8', 1),
      ('hrs_h_filler4', '>i2', 5),
      ('hrs_h_autocalind', '>i2', 1),
      ('hrs_h_solarcalyr', '>i2', 1),
      ('hrs_h_solarcaldy', '>i2', 1),
      ('hrs_h_calinf', '>i4', 80),
      ('hrs_h_filler5', '>i4', 2),
      ('hrs_h_tempradcnv', '>i4', 57),
      ('hrs_h_20solfiltirrad', '>i2', 1),
      ('hrs_h_20equifiltwidth', '>i2', 1),
      # CORRECTION! NWPSAF guide says there is 1 field
      # here, but in reality it is 2 (see NOAA KLM User's
      # Guide, page 8-110, PDF page 421)
      ('hrs_h_filler6', '>i4', 2),
      ('hrs_h_modelid', '|S8', 1),
      ('hrs_h_nadloctol', '>i2', 1),
      ('hrs_h_locbit', '>i2', 1),
      ('hrs_h_filler7', '>i2', 1),
      ('hrs_h_rollerr', '>i2', 1),
      ('hrs_h_pitcherr', '>i2', 1),
      ('hrs_h_yawerr', '>i2', 1),
      ('hrs_h_epoyr', '>i2', 1),
      ('hrs_h_epody', '>i2', 1),
      ('hrs_h_epotime', '>i4', 1),
      ('hrs_h_smaxis', '>i4', 1),
      ('hrs_h_eccen', '>i4', 1),
      ('hrs_h_incli', '>i4', 1),
      ('hrs_h_argper', '>i4', 1),
      ('hrs_h_rascnod', '>i4', 1),
      ('hrs_h_manom', '>i4', 1),
      ('hrs_h_xpos', '>i4', 1),
      ('hrs_h_ypos', '>i4', 1),
      ('hrs_h_zpos', '>i4', 1),
      ('hrs_h_xvel', '>i4', 1),
      ('hrs_h_yvel', '>i4', 1),
      ('hrs_h_zvel', '>i4', 1),
      ('hrs_h_earthsun', '>i4', 1),
      ('hrs_h_filler8', '>i4', 4),
      ('hrs_h_rdtemp', '>i2', 6),
      ('hrs_h_bptemp', '>i2', 6),
      ('hrs_h_eltemp', '>i2', 6),
      ('hrs_h_pchtemp', '>i2', 6),
      ('hrs_h_fhcc', '>i2', 6),
      ('hrs_h_scnmtemp', '>i2', 6),
      ('hrs_h_fwmtemp', '>i2', 6),
      ('hrs_h_p5v', '>i2', 6),
      ('hrs_h_p10v', '>i2', 6),
      ('hrs_h_p75v', '>i2', 6),
      ('hrs_h_m75v', '>i2', 6),
      ('hrs_h_p15v', '>i2', 6),
      ('hrs_h_m15v', '>i2', 6),
      ('hrs_h_fwmcur', '>i2', 6),
      ('hrs_h_scmcur', '>i2', 6),
      ('hrs_h_pchcpow', '>i2', 6),
      # CORRECTION: Due to the earlier error, there's 889
      # left, not 890, for the total itemsize must remain
      # 4608
      ('hrs_h_filler9', '>i4', 890)])

HIRS_line_dtypes[3] = numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlinyr', '>i2', 1),
      ('hrs_scnlindy', '>i2', 1),
      ('hrs_clockdrift', '>i2', 1),
      ('hrs_scnlintime', '>i4', 1),
      ('hrs_scnlinf', '>i2', 1),
      ('hrs_mjfrcnt', '>i2', 1),
      ('hrs_scnpos', '>i2', 1),
      ('hrs_scntyp', '>i2', 1),
      ('hrs_filler1', '>i4', 2),
      ('hrs_qualind', '>i4', 1),
      ('hrs_linqualflgs', '>i4', 1),
      ('hrs_chqualflg', '>i2', 20),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler2', '>i4', 4),
      ('hrs_calcof', '>i4', 60),
      ('hrs_scalcof', '>i4', 60),
      ('hrs_filler3', '>i4', 3),
      ('hrs_navstat', '>i4', 1),
      ('hrs_attangtime', '>i4', 1),
      ('hrs_rollang', '>i2', 1),
      ('hrs_pitchang', '>i2', 1),
      ('hrs_yawang', '>i2', 1),
      ('hrs_scalti', '>i2', 1),
      ('hrs_ang', '>i2', 168),
      ('hrs_pos', '>i4', 112),
      ('hrs_filler4', '>i4', 2),
      ('hrs_elem', '>i2', 1536),
      ('hrs_filler5', '>i4', 3),
      ('hrs_digbinvwbf', '>i2', 1),
      ('hrs_digitbwrd', '>i2', 1),
      ('hrs_aninvwbf', '>i4', 1),
      ('hrs_anwrd', '>i1', 16),
      ('hrs_filler6', '>i4', 11)])

HIRS_header_dtypes[4] = numpy.dtype([('hrs_h_siteid', '|S3', 1),
      ('hrs_h_blank', '|S1', 1),
      ('hrs_h_l1bversnb', '>i2', 1),
      ('hrs_h_l1bversyr', '>i2', 1),
      ('hrs_h_l1bversdy', '>i2', 1),
      ('hrs_h_reclg', '>i2', 1),
      ('hrs_h_blksz', '>i2', 1),
      ('hrs_h_hdrcnt', '>i2', 1),
      ('hrs_h_filler0', '>i2', 3),
      ('hrs_h_dataname', '|S42', 1),
      ('hrs_h_prblkid', '|S8', 1),
      ('hrs_h_satid', '>i2', 1),
      ('hrs_h_instid', '>i2', 1),
      ('hrs_h_datatyp', '>i2', 1),
      ('hrs_h_tipsrc', '>i2', 1),
      ('hrs_h_startdatajd', '>i4', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_startdatady', '>i2', 1),
      ('hrs_h_startdatatime', '>i4', 1),
      ('hrs_h_enddatajd', '>i4', 1),
      ('hrs_h_enddatayr', '>i2', 1),
      ('hrs_h_enddatady', '>i2', 1),
      ('hrs_h_enddatatime', '>i4', 1),
      ('hrs_h_cpidsyr', '>i2', 1),
      ('hrs_h_cpidsdy', '>i2', 1),
      # CORRECTION! NWPSAF guide says there are 4 fields
      # here, but in reality there is 1 (see NOAA KLM
      # Users Guide – April 2014 Revision, page 8-117, PDF
      # page 428)
      ('hrs_h_fov1offset', '>i2', 1),
      ('hrs_h_instrtype', '|S6', 1),
      ('hrs_h_inststat1', '>i4', 1),
      ('hrs_h_filler1', '>i2', 1),
      ('hrs_h_statchrecnb', '>i2', 1),
      ('hrs_h_inststat2', '>i4', 1),
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_callocsclin', '>i2', 1),
      ('hrs_h_misscnlin', '>i2', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_okdatafr', '>i2', 1),
      ('hrs_h_pacsparityerr', '>i2', 1),
      ('hrs_h_auxsyncerrsum', '>i2', 1),
      ('hrs_h_timeseqerr', '>i2', 1),
      ('hrs_h_timeseqerrcode', '>i2', 1),
      ('hrs_h_socclockupind', '>i2', 1),
      ('hrs_h_locerrind', '>i2', 1),
      ('hrs_h_locerrcode', '>i2', 1),
      ('hrs_h_pacsstatfield', '>i2', 1),
      ('hrs_h_pacsdatasrc', '>i2', 1),
      ('hrs_h_filler2', '>i4', 1),
      ('hrs_h_spare1', '|S8', 1),
      ('hrs_h_spare2', '|S8', 1),
      ('hrs_h_filler3', '>i2', 5),
      ('hrs_h_autocalind', '>i2', 1),
      ('hrs_h_solarcalyr', '>i2', 1),
      ('hrs_h_solarcaldy', '>i2', 1),
      ('hrs_h_calinf', '>i4', 80),
      # CORRECTION! NWPSAF calls this hrs_h_filler5, which
      # already occurs a few lines down.
      ('hrs_h_filler4', '>i4', 2),
      ('hrs_h_tempradcnv', '>i4', 57),
      ('hrs_h_20solfiltirrad', '>i2', 1),
      ('hrs_h_20equifiltwidth', '>i2', 1),
      # CORRECTION! NWPSAF guide says there is 1 such
      # field, in reality there are 2.  See NOAA KLM
      # User's Guide, April 2014 Revision, Page 8-124 /
      # PDF Page 435
      ('hrs_h_filler5', '>i4', 2),
      ('hrs_h_modelid', '|S8', 1),
      ('hrs_h_nadloctol', '>i2', 1),
      ('hrs_h_locbit', '>i2', 1),
      ('hrs_h_filler6', '>i2', 1),
      ('hrs_h_rollerr', '>i2', 1),
      ('hrs_h_pitcherr', '>i2', 1),
      ('hrs_h_yawerr', '>i2', 1),
      ('hrs_h_epoyr', '>i2', 1),
      ('hrs_h_epody', '>i2', 1),
      ('hrs_h_epotime', '>i4', 1),
      ('hrs_h_smaxis', '>i4', 1),
      ('hrs_h_eccen', '>i4', 1),
      ('hrs_h_incli', '>i4', 1),
      ('hrs_h_argper', '>i4', 1),
      ('hrs_h_rascnod', '>i4', 1),
      ('hrs_h_manom', '>i4', 1),
      ('hrs_h_xpos', '>i4', 1),
      ('hrs_h_ypos', '>i4', 1),
      ('hrs_h_zpos', '>i4', 1),
      ('hrs_h_xvel', '>i4', 1),
      ('hrs_h_yvel', '>i4', 1),
      ('hrs_h_zvel', '>i4', 1),
      ('hrs_h_earthsun', '>i4', 1),
      ('hrs_h_filler7', '>i4', 4),
      ('hrs_h_rdtemp', '>i4', 6),
      ('hrs_h_bptemp', '>i4', 6),
      ('hrs_h_eltemp', '>i4', 6),
      ('hrs_h_pchtemp', '>i4', 6),
      ('hrs_h_fhcc', '>i4', 6),
      ('hrs_h_scnmtemp', '>i4', 6),
      ('hrs_h_fwmtemp', '>i4', 6),
      ('hrs_h_p5v', '>i4', 6),
      ('hrs_h_p10v', '>i4', 6),
      ('hrs_h_p75v', '>i4', 6),
      ('hrs_h_m75v', '>i4', 6),
      ('hrs_h_p15v', '>i4', 6),
      ('hrs_h_m15v', '>i4', 6),
      ('hrs_h_fwmcur', '>i4', 6),
      ('hrs_h_scmcur', '>i4', 6),
      ('hrs_h_pchcpow', '>i4', 6),
      ('hrs_h_iwtcnttmp', '>i4', 30),
      ('hrs_h_ictcnttmp', '>i4', 24),
      ('hrs_h_tttcnttmp', '>i4', 6),
      ('hrs_h_fwcnttmp', '>i4', 24),
      ('hrs_h_patchexpcnttmp', '>i4', 6),
      ('hrs_h_fsradcnttmp', '>i4', 6),
      ('hrs_h_scmircnttmp', '>i4', 6),
      ('hrs_h_pttcnttmp', '>i4', 6),
      ('hrs_h_sttcnttmp', '>i4', 6),
      ('hrs_h_bpcnttmp', '>i4', 6),
      ('hrs_h_electcnttmp', '>i4', 6),
      ('hrs_h_patchfcnttmp', '>i4', 6),
      ('hrs_h_scmotcnttmp', '>i4', 6),
      ('hrs_h_fwmcnttmp', '>i4', 6),
      ('hrs_h_chsgcnttmp', '>i4', 6),
      ('hrs_h_conversions', '>i4', 11),
      ('hrs_h_moonscnlin', '>i2', 1),
      ('hrs_h_moonthresh', '>i2', 1),
      ('hrs_h_avspcounts', '>i4', 20),
      ('hrs_h_startmanyr', '>i2', 1),
      ('hrs_h_startmandy', '>i2', 1),
      ('hrs_h_startmantime', '>i4', 1),
      ('hrs_h_endmanyr', '>i2', 1),
      ('hrs_h_endmandy', '>i2', 1),
      ('hrs_h_endmantime', '>i4', 1),
      ('hrs_h_deltav', '>i4', 3),
      ('hrs_h_mass', '>i4', 2),
      ('hrs_h_filler8', '>i2', 1302)])

HIRS_line_dtypes[4] = numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlinyr', '>i2', 1),
      ('hrs_scnlindy', '>i2', 1),
      ('hrs_clockdrift', '>i2', 1),
      ('hrs_scnlintime', '>i4', 1),
      ('hrs_scnlinf', '>i2', 1),
      ('hrs_mjfrcnt', '>i2', 1),
      ('hrs_scnpos', '>i2', 1),
      ('hrs_scntyp', '>i2', 1),
      ('hrs_filler1', '>i4', 2),
      ('hrs_qualind', '>i4', 1),
      ('hrs_linqualflgs', '>i4', 1),
      ('hrs_chqualflg', '>i2', 20),
      # Correction: Not |S1 but >i1
      #('hrs_mnfrqual', '|S1', 64),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler2', '>i4', 4),
      ('hrs_calcof', '>i4', 60),
      ('hrs_scalcof', '>i4', 60),
      ('hrs_yawsteering', '>i2', 3),
      ('hrs_totattcorr', '>i2', 3),
      ('hrs_navstat', '>i4', 1),
      ('hrs_attangtime', '>i4', 1),
      ('hrs_rollang', '>i2', 1),
      ('hrs_pitchang', '>i2', 1),
      ('hrs_yawang', '>i2', 1),
      ('hrs_scalti', '>i2', 1),
      ('hrs_ang', '>i2', 168),
      ('hrs_pos', '>i4', 112),
      ('hrs_moonang', '>i2', 1),
      # CORRECTION: NWPSAF formatting guide calls this
      # filler4.  Should be filler3.
      ('hrs_filler3', '>i2', 3),
      ('hrs_elem', '>i2', 1536),
      # CORRECTION: NWPSAF formatting guide calls this
      # filler5.  Should be filler4.
      ('hrs_filler4', '>i4', 3),
      ('hrs_digitbupdatefg', '>i2', 1),
      ('hrs_digitbwrd', '>i2', 1),
      ('hrs_analogupdatefg', '>i4', 1),
      # CORRECTION: |S1 does not make sense, read as >i1 instead
      ('hrs_anwrd', '>i1', 16),
      ('hrs_filler5', '>i4', 11)])

HIRS_channel_order = {}
HIRS_channel_order[2] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[3] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[4] = HIRS_channel_order[3].copy()

# obtained manually from POD User's Guide

# Source: POD User's Guide, Table 2.0.4-1.
HIRS_header_dtypes[2] = numpy.dtype([
      ("hrs_h_satid", ">i1", 1),
      ('hrs_h_datatyp', '>i1', 1),
      ('hrs_h_startdatadatetime', '|S6', 1), # read as bytes for now
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_enddatadatetime', '|S6', 1), # read as bytes for now  
      ('hrs_h_pbid', '|S7', 1),
      ('hrs_h_autocalind', '>i1', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_dacsqual', '>u2', 3),
      ('hrs_h_calid', '>u1', 2),
      ('hrs_h_dacsstat', '>i1', 1),
      ('hrs_h_attcorr', '>i1', 1),
      ('hrs_h_nadloctol', '>i1', 1),
      ('hrs_h_filler0', '>i1', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_dataname', '|S42', 1), # EBCDIC!
      ('hrs_h_filler7', '>i1', 3), # Not sure why
      ('hrs_h_filler8', '>i4', 1042)])

# Source: POD User's Guide, Section 4-1
HIRS_line_dtypes[2] = numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlintime', '|S6', 1), # read as bytes for now
      ('hrs_qualind', '>i4', 1),
      ('hrs_earthlocdelta', '>i4', 1),
      ('hrs_calcof', '>i4', 60*3),
      ('hrs_satloc', '>i2', 2),
      ('hrs_pos', '>i2', 112),
      ('hrs_elem', '>i2', 1408),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler0', '>i1', 409),
    ])

# For HIRS/3, conversion of counts to brightness temperatures for Digital
# A Telemetry is not included in the files, but partially included in the
# KLM User's Guide.  However, this includes only coefficients for the
# Internal Warm Target (IWT) Platinum Resistance Thermometer (PRT) and the
# secondary telescope.

HIRS_count_to_temp = {}

for sat in {"TIROSN", "NOAA6", "NOAA7", "NOAA8", "NOAA9", "NOAA10", "NOAA11", "NOAA12", "NOAA14", "NOAA15", "NOAA16", "NOAA17"}:
    HIRS_count_to_temp[sat] = {}

# Table D.1-2.

HIRS_count_to_temp["NOAA15"]["iwtcnttmp"] = numpy.array([
    [301.42859, 6.5398670E-03, 8.9808960E-08, 4.7877130E-11, 1.3453590E-15],
    [301.44106, 6.5306330E-03, 8.7115040E-08, 4.7387900E-11, 1.4460280E-15],
    [301.43252, 6.5332780E-03, 8.2485710E-08, 4.7301670E-11, 1.6099050E-15],
    [301.39868, 6.5244370E-03, 8.0380230E-08, 4.7093000E-11, 1.6976440E-15]])

# Table D.1-16

HIRS_count_to_temp["NOAA15"]["sttcnttmp"] = numpy.array([
    260.29119, 1.693469E-02, -2.413170E-06, 4.019185E-10, 1.175655E-14])

# Table D.2-2.

HIRS_count_to_temp["NOAA16"]["iwtcnttmp"] = numpy.array([
    [301.45076, 6.530210E-03, 8.326151E-08, 4.724724E-11, 1.565263E-15],
    [301.39565, 6.527550E-03, 8.417738E-08, 4.727738E-11, 1.460746E-15],
    [301.40733, 6.528222E-03, 8.314237E-08, 4.721744E-11, 1.543985E-15],
    [301.40280, 6.525508E-03, 8.269671E-08, 4.707211E-11, 1.549894E-15]])

# Table D.2-5

HIRS_count_to_temp["NOAA16"]["sttcnttmp"] = numpy.array([
    260.42546, 1.659977E-02, -2.118035E-06, 3.040075E-10, 2.251628E-14])

# Table D.3-11

HIRS_count_to_temp["NOAA17"]["iwtcnttmp"] = numpy.array([
    [301.41859, 0.006539867, 8.909E-08, 4.78771E-11, 1.34536E-15],
    [301.43106, 0.006530633, 8.7115E-08, 4.73879E-11, 1.44603E-15],
    [301.42252, 0.006533278, 8.24857E-08, 4.73017E-11, 1.60991E-15],
    [301.38868, 0.006524437, 8.03802E-08, 4.7093E-11, 1.69764E-15]])

# Table D.3-12

HIRS_count_to_temp["NOAA17"]["sttcnttmp"] = numpy.array([
    260.29119, 0.01693469, -2.41317E-06, 4.01919E-10, 1.17566E-14])

# Remaining information for NOAA-15 onward is based on CPIDS information
# sent by Dejiang Han <dejiang.han@noaa.gov> to Gerrit Holl
# <g.holl@reading.ac.uk> on 2016-02-17.




# For HIRS/2, the POD guide does not appear to include any coefficients.
# Information is scarce.

# Source: Levin Gary, J Nelson, Frank W Porto, Data Extraction and
# calibration of TIROS-N/NOAA radiometers, NOAA Technical Memorandum NESS
# 107, Appendix B http://docs.lib.noaa.gov/rescue/TIROS/QC8795U4no107.pdf
#
# "This document contains Appendix B for TIROS-N, NOAA-9 and NOAA-10.
# Appendix B for other spacecraft will be issued separately."

# PDF page 77
#HIRS_count_to_temp["TIROSN"]["iwtcnttmp"] = "FIXME"

# PDF page 92, NOAA F/9
#HIRS_count_to_temp["NOAA9"]["iwtcnttmp"] = "FIXME"

# PDF page 104, NOAA G/10
#HIRS_count_to_temp["NOAA10"]["iwtcnttmp"] = "FIXME"

# NOAA H/11
# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal11.asc
#HIRS_count_to_temp["NOAA11"]["iwtcnttmp"] = "FIXME"

# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal12.asc

# http://www.sat.dundee.ac.uk/noaa14.html

# Fill what's missing with dummies
dummy = numpy.ma.array([numpy.ma.masked])
for sat in {"NOAA6", "NOAA7", "NOAA8", "NOAA9", "NOAA10", "NOAA11",
            "NOAA12", "NOAA14", "NOAA15", "NOAA16", "NOAA17"}:
    for field in {"fwcnttmp", "patchexpcnttmp", "fsradcnttmp",
                  "scmircnttmp", "pttcnttmp", "sttcnttmp", "bpcnttmp",
                  "electcnttmp", "patchfcnttmp", "scmotcnttmp",
                  "fwmcnttmp", "chsgcnttmp", "iwtcnttmp"}:
        if not field in HIRS_count_to_temp[sat]:
            # When it's (n, 6) for HIRS/4, it's (n, 5) for HIRS/3 and
            # HIRS/2
            HIRS_count_to_temp[sat][field] = numpy.tile(dummy,
                    (HIRS_header_dtypes[4]["hrs_h_"+field].shape[0]//6, 5))
    if not "ictcnttmp" in HIRS_count_to_temp[sat]:
        # but this one is (4, 5) on HIRS/2 and HIRS/3
        HIRS_count_to_temp[sat]["ictcnttmp"] = numpy.tile(dummy, (4, 5))

# For HIRS/2, central wavenumbers and coefficients for BT conversion are
# not included in the headers.  Include them here.  Taken from Nick
# Bearsons HIRStoHDF code at
# https://svn.ssec.wisc.edu/repos/HIRStoHDF/trunk/src/HTH_HIRS2_inc.py
#
# FIXME: verify agreement with NOAA POD Guide

HIRS_coeffs = {
    2: {
        14:
       { 1 : ( 668.90 ,  0.002  , 0.99998 ),
         2 : ( 679.36 ,  -0.000 , 0.99997 ),
         3 : ( 689.63 ,  0.011  , 0.99994 ),
         4 : ( 703.56 ,  0.001  , 0.99994 ),
         5 : ( 714.50 ,  -0.014 , 0.99997 ),
         6 : ( 732.28 ,  0.026  , 0.99989 ),
         7 : ( 749.64 ,  0.019  , 0.99991 ),
         8 : ( 898.67 ,  0.067  , 0.99977 ),
         9 : ( 1028.31,  0.050  , 0.99980 ),
        10 : ( 796.04 ,  0.021  , 0.99990 ),
        11 : ( 1360.95,  0.073  , 0.99971 ),
        12 : ( 1481.00,  0.284  , 0.99931 ),
        13 : ( 2191.32,  0.021  , 0.99996 ),
        14 : ( 2207.36,  0.020  , 0.99997 ),
        15 : ( 2236.39,  0.024  , 0.99998 ),
        16 : ( 2268.12,  0.018  , 0.99996 ),
        17 : ( 2420.24,  0.026  , 0.99992 ),
        18 : ( 2512.21,  0.042  , 0.99993 ),
        19 : ( 2647.91,  0.313  , 0.99946 )},

        13:
       { 1 :  ( 668.81 , -0.077,  1.00019 ),
         2  : ( 679.59 , 0.020 ,  0.99992 ),
         3  : ( 690.18 , 0.016 ,  0.99993 ),
         4  : ( 703.02 , 0.018 ,  0.99991 ),
         5  : ( 715.96 , 0.040 ,  0.99986 ),
         6  : ( 732.98 , 0.028 ,  0.99987 ),
         7  : ( 749.34 , -0.034,  1.00000 ),
         8  : ( 902.39 , 0.544 ,  0.99916 ),
         9  : ( 1028.77, 0.062 ,  0.99979 ),
         10 : ( 792.82 , -0.005,  0.99994 ),
         11 : ( 1359.95, 0.090 ,  0.99972 ),
         12 : ( 1479.90, 0.292 ,  0.99931 ),
         13 : ( 2189.06, 0.022 ,  0.99997 ),
         14 : ( 2212.55, 0.021 ,  0.99997 ),
         15 : ( 2231.68, 0.029 ,  0.99993 ),
         16 : ( 2267.04, 0.022 ,  0.99999 ),
         17 : ( 2418.31, 0.025 ,  0.99992 ),
         18 : ( 2516.80, 0.058 ,  0.99970 ),
         19 : ( 2653.33, 0.264 ,  0.99927 )},

        12:
       { 1  : ( 667.58 ,    0.007 ,   0.99996),
         2  : ( 680.18 ,    0.007 ,   0.99995),
         3  : ( 690.01 ,    0.019 ,   0.99989),
         4  : ( 704.22 ,    0.026 ,   0.99988),
         5  : ( 716.32 ,    0.021 ,   0.99990),
         6  : ( 732.81 ,    0.140 ,   0.99964),
         7  : ( 751.92 ,    0.058 ,   0.99982),
         8  : ( 900.45 ,    0.358 ,   0.99940),
         9  : ( 1026.66,    0.181 ,   0.99985),                       
         10 : ( 1223.44,    0.377 ,   0.99975),
         11 : ( 1368.68,    0.175 ,   0.99992),
         12 : ( 1478.59,    0.265 ,   0.99863),
         13 : ( 2190.37,    0.078 ,   1.00042),
         14 : ( 2210.51,    0.017 ,   0.99995),
         15 : ( 2236.62,    -0.023,   0.99950),
         16 : ( 2267.62,    0.021 ,   0.99995),
         17 : ( 2361.64,    0.022 ,   0.99997),
         18 : ( 2514.68,    0.058 ,   0.99992),
         19 : ( 2653.48,    0.344 ,   0.99950)},

        11:
       { 1  : ( 668.99 ,    0.007 ,   0.99996),
         2  : ( 678.89 ,    0.010 ,   0.99994),
         3  : ( 689.70 ,    0.007 ,   0.99992),
         4  : ( 703.25 ,    -0.003,   0.99995),
         5  : ( 716.83 ,    0.014 ,   0.99991),
         6  : ( 732.11 ,    0.019 ,   0.99991),
         7  : ( 749.48 ,    0.032 ,   0.99988),
         8  : ( 900.51 ,    0.077 ,   0.99988),                       
         9  : ( 1031.19,    0.068 ,   0.99975),
         10 : ( 795.69 ,    -0.001,   0.99994),
         11 : ( 1361.10,    0.074 ,   0.99972),
         12 : ( 1479.86,    0.288 ,   0.99994),
         13 : ( 2189.94,    0.022 ,   0.99994),
         14 : ( 2209.66,    0.018 ,   0.99995),
         15 : ( 2239.26,    0.020 ,   0.99995),
         16 : ( 2267.80,    0.015 ,   0.99993),
         17 : ( 2416.32,    0.024 ,   0.99991),
         18 : ( 2511.83,    0.045 ,   0.99990),
         19 : ( 2664.07,    0.325 ,   0.99949)},
         
         10:   
       { 1  : ( 667.70 ,    0.033 ,   0.99989),
         2  : ( 680.23 ,    0.018 ,   0.99992),
         3  : ( 691.15 ,    -0.006,   0.99994),
         4  : ( 704.33 ,    -0.002,   0.99994),
         5  : ( 716.30 ,    -0.064,   1.00007),
         6  : ( 733.13 ,    0.065 ,   0.99980),
         7  : ( 750.72 ,    0.073 ,   0.99979),
         8  : ( 899.50 ,    0.218 ,   0.99957),
         9  : ( 1029.01,    0.195 ,   0.99987),                       
         10 : ( 1224.07,    0.327 ,   0.99965),
         11 : ( 1363.32,    0.046 ,   0.99963),
         12 : ( 1489.42,    0.645 ,   1.00064),
         13 : ( 2191.38,    0.072 ,   1.00036),
         14 : ( 2208.74,    0.079 ,   1.00045),
         15 : ( 2237.49,    -0.026,   0.99947),
         16 : ( 2269.09,    0.041 ,   1.00019),
         17 : ( 2360.00,    0.040 ,   1.00019),
         18 : ( 2514.58,    0.098 ,   1.00025),
         19 : ( 2665.38,    0.462 ,   1.00067) },
         
        9:
       { 1  : ( 667.67 ,    0.034 ,   0.99989),
         2  : ( 679.84 ,    0.024 ,   0.99991),
         3  : ( 691.46 ,    0.092 ,   0.99975),
         4  : ( 703.37 ,    0.002 ,   0.99993),
         5  : ( 717.16 ,    0.013 ,   0.99991),
         6  : ( 732.64 ,    -0.023,   0.99997),
         7  : ( 749.48 ,    -0.006,   0.99995),
         8  : ( 898.53 ,    0.126 ,   0.99969),                       
         9  : ( 1031.61,    0.187 ,   0.99987),
         10 : ( 1224.74,    0.569 ,   1.00010),
         11 : ( 1365.12,    0.033 ,   0.99961),
         12 : ( 1483.24,    0.353 ,   0.99911),
         13 : ( 2189.97,    -0.001,   0.99980),
         14 : ( 2209.18,    0.007 ,   0.99984),
         15 : ( 2243.14,    0.027 ,   1.00003),
         16 : ( 2276.46,    0.099 ,   1.00038),
         17 : ( 2359.05,    0.004 ,   0.99977),
         18 : ( 2518.14,    0.084 ,   1.00012),
         19 : ( 2667.80,    0.448 ,   1.00040)},
         
        
        8:
       { 1  : ( 667.41 ,    0.099 ,   0.99971),
         2  : ( 679.45 ,    0.147 ,   0.99962),
         3  : ( 690.90 ,    0.143 ,   0.99964),
         4  : ( 702.97 ,    0.010 ,   0.99991),
         5  : ( 717.56 ,    -0.001,   0.99994),
         6  : ( 732.97 ,    0.193 ,   0.99955),
         7  : ( 747.90 ,    -0.104,   1.00013),
         8  : ( 901.08 ,    0.429 ,   0.99931),
         9  : ( 1027.11,    0.140 ,   0.99984),                       
         10 : ( 1224.05,    0.450 ,   0.99988),
         11 : ( 1366.17,    0.108 ,   0.99978),
         12 : ( 1486.92,    0.530 ,   1.00008),
         13 : ( 2189.28,    0.051 ,   1.00022),
         14 : ( 2211.71,    0.063 ,   1.00029),
         15 : ( 2238.06,    0.015 ,   0.99992),
         16 : ( 2271.43,    0.029 ,   1.00004),
         17 : ( 2357.11,    0.018 ,   0.99993),
         18 : ( 2515.53,    0.080 ,   1.00007),
         19 : ( 2661.85,    0.489 ,   1.00061)},
        
        7:
       { 1  : ( 667.92 ,    -0.010,   1.00001),
         2  : ( 679.21 ,    0.100 ,   0.99973),
         3  : ( 691.56 ,    -0.018,   0.99997),
         4  : ( 704.63 ,    0.026 ,   0.99989),
         5  : ( 717.05 ,    -0.009,   0.99995),
         6  : ( 733.20 ,    -0.081,   1.00008),
         7  : ( 749.20 ,    -0.054,   1.00003),                       
         8  : ( 898.94 ,    0.332 ,   0.99942),
         9  : ( 1027.38,    0.205 ,   0.99987),
         10 : ( 1224.89,    0.469 ,   0.99994),
         11 : ( 1363.85,    0.114 ,   0.99983),
         12 : ( 1489.06,    0.573 ,   1.00028),
         13 : ( 2183.05,    0.047 ,   1.00013),
         14 : ( 2208.28,    0.060 ,   1.00028),
         15 : ( 2239.84,    0.021 ,   0.99993),
         16 : ( 2271.33,    0.032 ,   1.00008),
         17 : ( 2357.55,    0.032 ,   1.00005),
         18 : ( 2512.83,    0.026 ,   0.99968),
         19 : ( 2663.79,    0.637 ,   1.00171)},
         
        6:
       { 1  : ( 668.02 ,    0.025 ,   0.99992),
         2  : ( 679.94 ,    0.151 ,   0.99900),
         3  : ( 690.44 ,    0.115 ,   0.99970),
         4  : ( 704.69 ,    0.041 ,   0.99984),
         5  : ( 717.43 ,    -0.035,   1.00000),
         6  : ( 732.47 ,    0.066 ,   0.99980),
         7  : ( 748.48 ,    -0.101,   1.00012),
         8  : ( 900.64 ,    0.185 ,   0.99961),
         9  : ( 1029.48,    0.268 ,   0.99990),                       
         10 : ( 1217.77,    -0.205,   0.99877),
         11 : ( 1368.05,    0.073 ,   0.99966),
         12 : ( 1485.76,    0.597 ,   1.00026),
         13 : ( 2190.60,    0.022 ,   1.00000),
         14 : ( 2210.09,    -0.001,   0.99978),
         15 : ( 2237.76,    0.029 ,   0.99999),
         16 : ( 2269.43,    0.015 ,   0.99991),
         17 : ( 2360.42,    0.011 ,   0.99984),
         18 : ( 2514.97,    0.051 ,   0.99985),
         19 : ( 2654.58,    0.482 ,   1.00042)}
     } }

HIRS_ids = {
    2: {
        4: 7,
        6: 8,
        7: 9,
        8: 10,
        1: 11, # Warning: identical to TIROS-N
        5: 12,
        2: 6, # Warning: identical to NOAA-13
        3: 14
    },
    3: {
        2: 16,
        4: 15,
        6: 17},
    4: {
        7: 18,
        8: 19,
        11: "A", # MetOp-A
        12: "B", # MetOp-B
        13: "C"}
}

HIRS_names = {
    2: {
        4: "NOAA7",
        6: "NOAA8",
        7: "NOAA9",
        8: "NOAA10",
        1: "NOAA11",
        5: "NOAA12",
        2: "NOAA13",
        3: "NOAA14"
    },
    3: {
        2: "NOAA16",
        4: "NOAA15",
        6: "NOAA17"},
    4: {
        7: "NOAA18",
        8: "NOAA19",
        11: "METOPA", # MetOp-A
        12: "METOPB", # MetOp-B
        13: "METOPC"}
}


# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal11.asc

# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal12.asc

# http://www.sat.dundee.ac.uk/noaa14.html

# NOAA POLAR SATELLITE CALIBRATION: A SYSTEM DESCRIPTION,
#   NOAA Technical Report NESDIS 77
# http://docs.lib.noaa.gov/rescue/TIROS/QC8795U47no77.pdf

# DATA EXTRACTION AND CALIBRATION OF TIROS-N/NOAA RADIOMETER
#   NOAA Technical Memorandum NESS 107
# http://docs.lib.noaa.gov/rescue/TIROS/QC8795U4no107.pdf
