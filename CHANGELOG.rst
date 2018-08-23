*********
Changelog
*********

v0.2.9 (2018-08-23)
===================
- Following changes in MOT.


v0.2.8 (2018-08-17)
===================
- Removed redundant super arguments.
- Following changes in MOT and MDT.


v0.2.7 (2018-08-02)
===================
- Regression fix.


v0.2.6 (2018-08-02)
===================
- Following changes in MOT.
- Removed six as a dependency.


v0.2.5 (2018-07-17)
===================
- Updated makefile to use twine for uploading to PyPi.


v0.2.4 (2018-05-03)
===================
- Following changes in MOT.


v0.2.3 (2018-04-11)
===================
- Following changes in MOT.


v0.2.2 (2018-04-09)
===================
- Following changes in MOT and MDT.


v0.2.1 (2018-04-04)
===================

Added
-----
- Adds post-optimization transformation to STARC.

Changed
-------
- Updates following the changes in MOT.

Other
-----
- Merge branch 'master' of github.com:cbclab/MCT.
- Version bump.
- Following changes in MOT.


v0.2 (2017-09-22)
=================

Added
-----
- Adds rCovSoS, improvements to rSoS. Adds calculation method for calculating the noise covariance matrix.
- Adds the rCovSoS method.
- Adds a split volume script to split single volumes into channels.

Changed
-------
- Updates to the comments.
- Changed the whole restructiring pipelines to make room for GRAPPA and SENSE.

Other
-----
- Prepared next release. Added a volumes switch to the reconstruction method.
- Small refactoring in the API, made the channels a constructor parameter.
- Merge branch 'master' of https://github.com/cbclab/MCT.
- Reverted the data loading again. This version uses more memory.
- Processing version with different kind of memory loading.
- Removed the multiprocessing in favor of single threaded data loading. It is more robust.
- Project rename.


v0.1.1 (2017-09-12)
===================

Added
-----
- Adds cl-device index flag and max batch size flag to the mct-reconstruct CLI.

Other
-----
- Prepared next release.


v0.1.0 (2017-09-12)
===================

Added
-----
- Adds changelog.

Changed
-------
- Updates to docs.
- Updates to docs.
- Updates to docs.
- Updates to docs.
- Updates to readme.

Other
-----
- Small update to docs.
- Small update to docs.
- Prepared for first release.
- First public release.


