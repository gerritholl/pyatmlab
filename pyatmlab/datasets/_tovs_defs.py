"""Relevant definitions for TOVS
"""

import numpy

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

HIRS_scale_factors[3] = dict(
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
    hrs_pos = 4)

HIRS_scale_factors[4] = HIRS_scale_factors[3].copy()

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
      ('hrs_anwrd', '|S1', 16),
      ('hrs_filler5', '>i4', 11)])

HIRS_channel_order = {}
HIRS_channel_order[2] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[3] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[4] = HIRS_channel_order[3].copy()

# obtained manually from POD User's Guide

# Source: KLM User's Guide, Section 2.0
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
      ('hrs_h_filler1', '>i2', 1),
      ('hrs_h_filler8', '>i4', 1043)])

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
      ('hrs_h_filler0', '>i1', 412),
    ])
