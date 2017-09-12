###############################
Maastricht Coil-combine Toolbox
###############################
The Maastricht Coil-combine Toolbox, MCT, is a small toolbox for combining the channels of a multi-channel MRI acquisition.
Where possible, this toolbox uses GPU accelerated routines to speed-up the processing.
For example, the weights of the STARC (STAbility-weighted Rf-coil Combination) reconstruction model are fitted using the GPU or using multi-threaded CPU.
At the moment MCT only supports rSoS (root Sum Of Squares) and STARC reconstruction, with plans for adding rCovSoS and others.


*******
Summary
*******
* GPU/multicore-CPU accelerated STARC
* rSoS (root Sum of Squares) and STARC available
* command line and python interface
* Free Open Source Software: LGPL v3 license
* Python and OpenCL based
* Full documentation: http://mct.readthedocs.io
* Project home: https://github.com/cbclab/MCT
* Uses the `GitLab workflow <https://docs.gitlab.com/ee/workflow/gitlab_flow.html>`_
* Tags: MRI, coil-combine, image reconstruction, opencl, python


*************
Fitting STARC
*************
You can use the following command for combining all your channels using the STARC reconstruction method:

.. code-block:: console

    $


************************
Quick installation guide
************************
The basic requirements for MCT are:

* Python 3.x (recommended) or Python 2.7
* OpenCL 1.2 (or higher) support in GPU driver or CPU runtime


**Linux**

For Ubuntu >= 16 you can use:

* ``sudo add-apt-repository ppa:robbert-harms/cbclab``
* ``sudo apt-get update``
* ``sudo apt-get install python3-mct``


For Debian users and Ubuntu < 16 users, install MDT with:

* ``sudo apt-get install python3 python3-pip python3-pyopencl python3-numpy python3-nibabel python3-pyqt5 python3-matplotlib python3-six python3-yaml python3-argcomplete libpng-dev libfreetype6-dev libxft-dev``
* ``sudo pip3 install mct``

Note that ``python3-nibabel`` may need NeuroDebian to be available on your machine. An alternative is to use ``pip3 install nibabel`` instead.


**Windows**

The installation on Windows is a little bit more complex and the following is only a quick reference guide.
To save duplication of information and since this package depends on MDT and MOT, the complete install instructions can be copied from
`the MDT documentation <https://maastrichtdiffusiontoolbox.readthedocs.org>`_.
After following that guide, installation of MCT is simply done using ``pip install mct``.
The quick overview is:

* Install Anaconda Python 3.5
* Install MOT using the guide at https://mot.readthedocs.io
* Open an Anaconda shell and type: ``pip install mct``


**Mac**

* Install Anaconda Python 3.5
* Open a terminal and type: ``pip install mct``

Please note that Mac support is experimental due to the unstable nature of the OpenCL drivers in Mac, that is, users running MDT with the GPU as selected device may experience crashes.
Running MDT in the CPU seems to work though.


For more information and full installation instructions please see the documentation of the MDT package https://maastrichtdiffusiontoolbox.readthedocs.org
