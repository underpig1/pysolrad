import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Optional

@njit
def gyrotaus(v, s, ne, temp, theta_rad, dl):
    """
    Computes gyroresonance optical depths for X and O modes at a given harmonic.

    Parameters
    ----------
    v : float
        Wave frequency [Hz].
    s : int
        Harmonic number (1 to 8).
    ne : float
        Electron number density [cm^-3].
    temp : float
        Electron temperature [K].
    theta_rad : float
        Angle between wavevector and magnetic field [radians].
    dl : float
        Path length element [cm].

    Returns
    -------
    taux : float
        Optical depth for X-mode.
    tauo : float
        Optical depth for O-mode.
    """

    if s < 1 or s > 8:
        return 0.0, 0.0
    me = 9.11e-28 # g
    kB = 1.38e-16 # erg/K
    cc = 2.99792458e10 # cm/s
    bb = v / (2.8e6 * s)  # Gauss
    fgyro = 2.8e6 * bb # Hz
    fplasma = 8980.0 * np.sqrt(ne)  # Hz

    Y = fgyro / v
    X = (fplasma / v)**2
    delta = np.sqrt((Y * np.sin(theta_rad))**4/4.0 + (1.0 - X)**2 * (Y*np.cos(theta_rad))**2)
    calc_tee = lambda dsgn: (-(Y * np.sin(theta_rad))**2 / 2.0 - dsgn*delta) / (Y * (1.0 - X)*np.cos(theta_rad))

    coeff = (np.pi**2 / 2.0) * 8.064e7 * ne * dl / v / cc
    factorials = (1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0)
    base_factor = coeff * s**2 / factorials[s - 1] * (s**2 * np.sin(theta_rad)**2 * kB * temp / (2.0 * me * cc**2))**(s - 1)

    calc_optical_depth = lambda tee: base_factor * ((1. + tee * np.cos(theta_rad))**2 + (temp / (5.9413e9 * tee))**2) / (1. + tee**2)
    calc_gyro = lambda mode: calc_optical_depth(calc_tee(-1.0 if mode == 'x' else 1.0))

    taux = calc_gyro('x') if s != 1 else 0.0
    tauo = calc_gyro('o')
    return taux, tauo

@njit
def taus(blos, btot, ne, Te, dl, v, dogyro=False):
    """
    Calculates differential optical depth along a ray segment.

    Includes both free-free bremsstrahlung and optionally gyroresonance contributions.

    Parameters
    ----------
    blos : float
        Line-of-sight magnetic field component [G].
    btot : float
        Total magnetic field strength [G].
    ne : float
        Electron number density [cm^-3].
    Te : float
        Electron temperature [K].
    dl : float
        Path length element [cm].
    v : float
        Wave frequency [Hz].
    dogyro : bool, optional
        Whether to include gyroresonance absorption (default False).

    Returns
    -------
    dtaur : float
        Right-circular polarization (RCP) optical depth increment.
    dtaul : float
        Left-circular polarization (LCP) optical depth increment.
    """

    f = 2.8e6 * blos/v
    # f = max(min(f, 0.99), 0.01)
    dtau = 0.2 * ne**2/v**2*Te**(-3./2.)*dl
    dtaur, dtaul = dtau/(1.-f)**2., dtau/(1.+f)**2.
    if btot > 0 and dogyro:
        s_eff = v / (2.8e6 * btot)
        s = int(np.round(s_eff))
        if 1 <= s <= 8 and np.abs(s - s_eff) < 0.2:
            theta_rad = np.arccos(min(max(blos / btot, -1.0), 1.0))
            taux, tauo = gyrotaus(v, s, ne, Te, theta_rad, dl)
            if blos < 0:
                taux, tauo = tauo, taux
            dtaur += taux
            dtaul += tauo
    return dtaur, dtaul

@njit
def radtrans(dtaur, dtaul, Te, Tbr, Tbl):
    """
    Applies radiative transfer update to brightness temperatures.

    Parameters
    ----------
    dtaur : float
        Incremental optical depth for RCP.
    dtaul : float
        Incremental optical depth for LCP.
    Te : float
        Electron temperature [K].
    Tbr : float
        Incoming brightness temperature in RCP [K].
    Tbl : float
        Incoming brightness temperature in LCP [K].

    Returns
    -------
    Tbr : float
        Updated RCP brightness temperature [K].
    Tbl : float
        Updated LCP brightness temperature [K].
    """
    Tbr, Tbl = Tbr*np.exp(-dtaur)+Te*(1.-np.exp(-dtaur)), Tbl*np.exp(-dtaul)+Te*(1.-np.exp(-dtaul))
    return Tbr, Tbl

@njit
def losint(ltemp, lbtot, lblos, lne, ldl, v, dogyro=False):
    """
    Does line-of-sight integration for a single ray.

    Integrates brightness temperatures, Faraday rotation, dispersion measure,
    and locates the optical depth unity surface.

    Parameters
    ----------
    ltemp : ndarray, shape (N,)
        Electron temperature [K] along the ray.
    lbtot : ndarray, shape (N,)
        Total magnetic field [G] along the ray.
    lblos : ndarray, shape (N,)
        Line-of-sight magnetic field component [G].
    lne : ndarray, shape (N,)
        Electron number density [cm^-3].
    ldl : ndarray, shape (N,)
        Path length elements [cm].
    v : float
        Wave frequency [Hz].
    dogyro : bool, optional
        Whether to include gyroresonance absorption (default False).

    Returns
    -------
    result : ndarray, shape (6,)
        Output values:
        - Stokes I [K]
        - Stokes V [K]
        - Faraday rotation [rad]
        - Dispersion measure [pc cm^-3]
        - Depth (distance) at optical unity surface [cm]
        - Optical unity index
    """
    npts = ltemp.shape[0]
    Tbr = Tbl = 3.0
    taur = taul = 0.0
    rm = disp = 0.0
    integrands = np.zeros((2, npts))
    for il in range(npts):
        ptemp = ltemp[il]
        pblos = lblos[il]
        pbtot = lbtot[il]
        pne = lne[il]
        pdl = ldl[il]
        if pdl == 0:
            continue
        ptemp = max(ptemp, 3)
        dtaur, dtaul = taus(pblos, pbtot, pne, ptemp, pdl, v, dogyro)
        integrands[:, il] = dtaur, dtaul
        Tbr, Tbl = radtrans(dtaur, dtaul, ptemp, Tbr, Tbl)
        rm += pne * pblos * pdl
        disp += pne*pdl
    I = (Tbr + Tbl) / 2.
    V = (Tbr - Tbl) / 2.
    faraday = 2.63e-13 * rm * (3e8 / v)**2
    unity = 0.0
    uidx = 0
    for uidx in range(npts - 1, -1, -1):
        dtaur, dtaul = integrands[:, uidx]
        taul += dtaul
        taur += dtaur
        unity += ldl[uidx]
        if (taul + taur)/2 >= 1.0:
            break
    return np.array((I, V, faraday, disp, unity, uidx), dtype=np.float32)

@njit(parallel=True)
def apply_losint(ftemp, fbtot, fblos, fne, fdl, v, dogyro=False):
    """
    Vectorized and parallelized line-of-sight synthesis for multiple rays.

    Parameters
    ----------
    ftemp : ndarray, shape (R, L)
        Electron temperature [K] for all rays.
    fbtot : ndarray, shape (R, L)
        Total magnetic field [G] for all rays.
    fblos : ndarray, shape (R, L)
        Line-of-sight magnetic field component [G].
    fne : ndarray, shape (R, L)
        Electron number density [cm^-3].
    fdl : ndarray, shape (R, L)
        Path length elements [cm].
    v : float
        Wave frequency [Hz].
    dogyro : bool, optional
        Whether to include gyroresonance absorption (default False).

    Returns
    -------
    out : ndarray, shape (R, 6)
        Synthesized outputs for all rays:
        - Stokes I [K]
        - Stokes V [K]
        - Faraday rotation [rad]
        - Dispersion measure [pc cm^-3]
        - Optical unity location [cm]
        - Optical unity index
    """
    out = np.empty((ftemp.shape[0], 6), dtype=np.float32)
    for i in prange(ftemp.shape[0]):
        out[i] = losint(ftemp[i], fbtot[i], fblos[i], fne[i], fdl[i], v, dogyro)
    return out

kB = 1.38e-16 # erg/K
cc = 2.99792458e10 # cm/s

class Quantity(np.ndarray):
    """
    A physical quantity that wraps a NumPy array with associated units and an optional frequency.

    Attributes
    ----------
    unit : str
        The unit of the quantity (e.g., 'K', 'cm', 'Jy/beam').
    v : float or None
        Frequency in Hz, required for certain conversions (e.g., brightness temperature <-> Jy/beam).

    Methods
    -------
    to(unit: str) -> Quantity
        Converts the quantity to a different unit.
    """
    _UNITS = {}
    _UNITS[('K', 'Jy/beam')] = lambda q: 2*kB*q/(cc/100/q.v)**2
    _UNITS[('Jy/beam', 'K')] = lambda q: q * (cc /100/ q.v) ** 2 / (2 * kB)
    _UNITS[('cm', 'm')] = lambda q: q/100
    _UNITS[('m', 'cm')] = lambda q: q*100
    _UNITS[('cm', 'Mm')] = lambda q: q*1e-8
    _UNITS[('Mm', 'cm')] = lambda q: q*1e8
    _UNITS[('m', 'Mm')] = lambda q: q*1e-6
    _UNITS[('Mm', 'm')] = lambda q: q*1e6
    _UNITS[('deg', 'rad')] = lambda q: np.deg2rad(q)
    _UNITS[('rad', 'deg')] = lambda q: np.rad2deg(q)
    _UNITS[('Hz', 'kHz')] = lambda q: q * 1e-3
    _UNITS[('kHz', 'Hz')] = lambda q: q * 1e3
    _UNITS[('Hz', 'MHz')] = lambda q: q * 1e-6
    _UNITS[('MHz', 'Hz')] = lambda q: q * 1e6
    _UNITS[('Hz', 'GHz')] = lambda q: q * 1e-9
    _UNITS[('GHz', 'Hz')] = lambda q: q * 1e9
    _UNITS[('kHz', 'MHz')] = lambda q: q * 1e-3
    _UNITS[('MHz', 'kHz')] = lambda q: q * 1e3
    _UNITS[('kHz', 'GHz')] = lambda q: q * 1e-6
    _UNITS[('GHz', 'kHz')] = lambda q: q * 1e6
    _UNITS[('MHz', 'GHz')] = lambda q: q * 1e-3
    _UNITS[('GHz', 'MHz')] = lambda q: q * 1e3

    def __new__(cls, arr, unit:str='cm', v:np.float32|float|None=None):
        obj = np.asarray(arr).view(cls)
        obj.unit = unit
        obj.v = v
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.unit = getattr(obj, 'unit', None)
        self.v = getattr(obj, 'v', None)

    def to(self, unit):
        """
        Convert the quantity to a different unit.

        Parameters
        ----------
        unit : str
            The target unit.

        Returns
        -------
        Quantity
            New quantity with converted units.

        Raises
        ------
        ValueError
            If the conversion is not supported or if frequency is required but not provided.
        """
        if (self.unit == unit):
            return self
        key = (self.unit, unit)
        if key not in self._UNITS:
            raise ValueError(f"No conversion from {key[0]} to {key[1]}")
        if self.v is None and unit == 'Jy/beam' or self.unit == 'Jy/beam':
            raise ValueError("Frequency 'v' was not provided")
        return Quantity(self._UNITS[key](self), unit, self.v)

    def __float__(self):
        if self.shape == ():
            return float(self.item())
        raise TypeError("Only scalar Quantity can be converted to float.")

    def __repr__(self):
        return f"{super().__repr__()} {self.unit}"

class Image:
    """
    Represents a synthesized image from radiative transfer, containing Stokes I, V, and other maps.

    Attributes
    ----------
    I : Quantity
        Stokes I intensity image in Kelvin.
    V : Quantity
        Stokes V (circular polarization) image in Kelvin.
    faraday : Quantity
        Faraday rotation angle in radians.
    disp : Quantity
        Dispersion measure in pc cm^-3.
    unity : Quantity
        Optical depth unity position in cm.
    shape : tuple
        Shape of the underlying data array.
    v : float
        Frequency at which the image was generated
    """
    def __init__(self, pixels: NDArray[np.float32], v=None):
        self.I = Quantity(pixels[..., 0], 'K', v)
        self.V = Quantity(pixels[..., 1], 'K', v)
        self.faraday = Quantity(pixels[..., 2], 'rad') 
        self.disp = Quantity(pixels[..., 3], 'pc cm^-3')
        self.unity = Quantity(pixels[..., 4], 'cm')
        self.uidx = pixels[..., 5]
        self.shape = pixels.shape
        self.v = v
    
    def reshape(self, *new_shape):
        self.I = self.I.reshape(new_shape)
        self.V = self.V.reshape(new_shape)
        self.faraday = self.faraday.reshape(new_shape)
        self.disp = self.disp.reshape(new_shape)
        self.unity = self.unity.reshape(new_shape)
        self.uidx = self.uidx.reshape(new_shape)
        self.shape = new_shape
        return self

class RayCollection:
    """
    Holds a collection of rays along which radiative transfer will be computed.

    Attributes
    ----------
    shape : tuple
        Original shape of the input 3D volumes.
    ftemp : ndarray
        Flattened temperature array, shape (N, L).
    fbtot : ndarray
        Flattened total magnetic field strength array, shape (N, L).
    fblos : ndarray
        Flattened line-of-sight magnetic field array, shape (N, L).
    fne : ndarray
        Flattened electron number density array, shape (N, L).
    fdl : ndarray
        Flattened path length array, shape (N, L).
    """
    def __init__(self, temp: NDArray[np.float32], btot: NDArray[np.float32], blos: NDArray[np.float32], ne: NDArray[np.float32], dl: np.float32 | float | NDArray[np.float32], axis:int=-1):
        """
        Initialize a collection of rays for synthesis.

        Parameters
        ----------
        temp : ndarray
            Electron temperature array.
        btot : ndarray
            Total magnetic field strength array.
        blos : ndarray
            Line-of-sight magnetic field strength array.
        ne : ndarray
            Electron number density array.
        dl : float or ndarray
            Path length per voxel along the integration direction.
        axis : int, default=-1
            Axis along which to integrate (e.g., line-of-sight).
        """
        shape = temp.shape
        assert btot.shape == shape, "Quantities have mismatched shapes"
        assert blos.shape == shape, "Quantities have mismatched shapes"
        assert ne.shape == shape, "Quantities have mismatched shapes"
        if not isinstance(dl, np.ndarray):
            dl = np.broadcast_to(dl, shape)
        else:
            assert dl.shape == shape, "Quantities have mismatched shapes"

        fray = lambda arr: self._unravel(np.ascontiguousarray(arr), axis)
        self.ftemp = fray(temp)
        self.fbtot = fray(btot)
        self.fblos = fray(blos)
        self.fne = fray(ne)
        self.fdl = fray(dl)
    
    def _unravel(self, arr, axis):
        axis = axis % arr.ndim
        arr_moved = np.moveaxis(arr, axis, -1)
        self.shape = arr_moved.shape
        leading = np.prod(self.shape[:-1], dtype=int)
        trailing = self.shape[-1]
        new_shape = (leading, trailing)
        return arr_moved.reshape(new_shape)

def synthesize(rays: RayCollection, v: np.float32 | float, dogyro:bool = False) -> Image:
    """
    Performs line-of-sight synthesis to compute a multi-channel image.

    Parameters
    ----------
    rays : RayCollection
        Input ray collection with all physical parameters.
    v : float
        Frequency in Hz at which the synthesis is performed.
    dogyro : bool, default=False
        Whether to include gyroresonance effects in the calculation.

    Returns
    -------
    Image
        A 2D image object containing Stokes I, V, Faraday rotation, dispersion, and optical unity position.
    """
    fimg_raw = apply_losint(rays.ftemp, rays.fbtot, rays.fblos, rays.fne, rays.fdl, v, dogyro)
    rimg = fimg_raw.reshape(*rays.shape[:-1], 6)
    return Image(rimg, v)