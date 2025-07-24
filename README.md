# Python SOLar RADio Coronal Emission Forward Synthesis Model

![](out/sunspot-side.png)
Figure 1: Generated maps from gyroresonance and bremsstrahlung coronal simulation of a synthetic MHD atmosphere, looking through the corona above a sunspot

> Abstract:<br>pySOLRAD forward synthesizes the dominant forms of solar coronal radio emission through a model solar atmosphere. The model simulates the interaction of particles throughout a synthetic solar atmosphere to produce bremsstrahlung and gyroresonance radio emission, and then ray-traces the absorption and emission mechanisms of those signals to the observer. As a result, several synthetic observables are generated, each representing a prediction of how solar radio signals would appear on Earth from a radio telescope. Because the properties of observed radio waves are physically dependent on the parameters of the model (eg. density, temperature, and magnetic field distributions), comparison of the outputted synthetic images with actual observations can be used to fine-tune these initial parameters and arrive at a much more complete understanding of the physical processes operating within the corona. The synthetic solar atmosphere passed to pySOLRAD is typically one generated from a magnetohydrodynamic (MHD) coronal simulation, which provides distributions for magnetic field, temperature, and density throughout the corona. The goal of pySOLRAD is to to improve our current understanding of the physical processes operating within the solar corona, especially in the less well-understood regime of solar radio physics.

# Features
- Support synthesis and simulation of both bremsstrahlung and gyroresonance radio emission (dominant forms of solar radio emission) through any given 1D, 2D, or 3D artifical solar atmosphere
- Generates maps, observables, and quantities for studying coronal intensity, circular/linear polarization, faraday rotation, dispersion measure, and optical depth
- Handles conversions between flux, flux density, intensity, frequency, polarization, and length units
- Well-documented with a wide range of examples

# Scripts
Create conda environment from cached `environment.yml`:
```batch
conda env create -f environment.yml
```

Update conda `environment.yml` (after installing a new dependency):
```batch
conda env export --from-history > environment.yml
```

Create conda environment from scratch (in case `environment.yml` is broken):
```batch
conda create -n pysolrad -c conda-forge python=3.13 numpy=2.2 numba=0.61 pyhdf pytables h5py sunpy
```

# File Descriptions
Notebooks:
* `1d-synthesis`: synthesize Stokes IV from a 1D representation of the solar environment, from Allen's Astrophysical Quantities
* `full-disk-synthesis`: synthesize Stokes IV and optical depth images from PSI's MAS model for the entire solar disk
* `sunspot-synthesis`: synthesize Stokes IV and optical depth images from MURaM model of a sunspot
* `norh-data`: generate Stokes IV images from real Nobeyama Radioheliograph data

Folders:
* `data`: data from Allen's Astrophysical Quantities, Nobeyama Radioheliograph, MURaM, and MAS
* `out`: sunspot and full-disk images from `sunspot-synthesis` and `full-disk-synthesis` runs

# Running the Notebooks
Most notebooks require a local download of the MHD models. Instructions for downloading these models is included [here](models/models.md). Then, create and activate a conda environment using the instructions above.