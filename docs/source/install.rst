=======
Install
=======

This section covers the basics of how to download and install `XFusion <https://github.com/xray-imaging/XFusion>`_.

.. contents:: Contents:
   :local:

Installing from source
======================

Install from `Anaconda <https://www.anaconda.com/distribution/>`_ > python3.9

Create and activate a dedicated conda environment::

    (base) $ conda create --name xfusion python=3.9
    (base) $ conda activate xfusion
    (xfusion) $ 

Clone the  `XFusion <https://github.com/xray-imaging/XFusion>`_ repository

::

    (xfusion) $ git clone https://github.com/xray-imaging/XFusion XFusion

Install XFusion::

    (xfusion) $ cd XFusion
    (xfusion) $ pip install .

Install all packages listed in the ``env/requirements.txt`` file::

    (xfusion) $ conda install numpy
    (xfusion) $ conda install add modules

Test the installation
=====================



Update
======

**XFusion** is constantly updated to include new features. To update your locally installed version::

    (xfusion) $ cd XFusion
    (xfusion) $ git pull
    (xfusion) $ pip install .


Dependencies
============

Install the following package::

    (xfusion) $ conda install numpy
    (xfusion) $ conda install add modules



