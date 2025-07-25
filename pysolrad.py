from typing import Literal
import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
import sys
import time
import threading

class Wheel:
    def __init__(self, text='working...', label="", total=None, delay=0.2):
        self.text = text
        self.chars = ['∙', '•', '●', '•']
        self.chars = ['/', '⎯', '\\', '|']
        self.label = label
        self.total = total
        self.delay = delay
        self.progress = 0
        self._lock = threading.Lock()
        self._index = 0
        self._running = False
        self._thread = None
    
    def _write(self, done=False):
        char = self.chars[self._index % len(self.chars)]
        if done:
            char = '⎯'
        suffix = f" {self.progress}/{self.total} {self.label}" if self.total is not None else ""
        sys.stdout.write(f"\r[{char}] - {self.text}{suffix}")
        sys.stdout.flush()
        if done:
            sys.stdout.write(f" - done")
            sys.stdout.flush()
            sys.stdout.write("\n")

    def _spin(self):
        while self._running:
            with self._lock:
                self._write()
                self._index += 1
            time.sleep(self.delay)

    def update(self):
        with self._lock:
            self.progress += 1

    def start(self):
        sys.stdout.write("\r")
        sys.stdout.flush()
        self._running = True
        self._thread = threading.Thread(target=self._spin)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        self._write(done = True)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __call__(self, iterable):
        self._iterable = iterable
        self.total = self.total or len(iterable) if hasattr(iterable, '__len__') else None
        self.start()
        return self

    def __iter__(self):
        if self._iterable is None:
            raise TypeError("Wheel must wrap an iterable to be used as an iterator.")
        for item in self._iterable:
            yield item
            self.update()
        self.stop()

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
    result : ndarray, shape (10,)
        Output values:
        - Stokes I [K]
        - Stokes V [K]
        - Faraday rotation [rad]
        - Dispersion measure [pc cm^-3]
        - Left-handed optical unity location [cm]
        - Right-handed optical unity location [cm]
        - Left-handed optical unity index
        - Right-handed optical unity index
        - Left-handed emissivity-weighted BLOS
        - Right-handed emissivity-weighted BLOS
    """
    npts = ltemp.shape[0]
    Tbr = Tbl = 3.0
    rm = disp = 0.0
    integrands = np.zeros((2, npts))
    cfl, cfr = 0.0, 0.0
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
    depth = 0.0
    unity_r, unity_l = 0.0, 0.0
    uidx_r, uidx_l = 0, 0
    taur = taul = 0.0
    num_l, denom_l, num_r, denom_r = 0.0, 0.0, 0.0, 0.0
    for il in range(npts - 1, -1, -1):
        ptemp = ltemp[il]
        pblos = lblos[il]
        dtaur, dtaul = integrands[:, il]
        taul += dtaul
        taur += dtaur
        depth += ldl[il]
        if taur >= 1.0 and unity_r == 0.0:
            unity_r = depth
            uidx_r = il
        if taul >= 1.0 and unity_l == 0.0:
            unity_l = depth
            uidx_l = il
        num_l += pblos*dtaul*ptemp*np.exp(-taul)
        denom_l += dtaul*ptemp*np.exp(-taul)
        num_r += pblos*dtaur*ptemp*np.exp(-taur)
        denom_r += dtaur*ptemp*np.exp(-taur)
    unity_r, unity_l = depth - unity_r, depth - unity_l
    weighted_blos_l = num_l/denom_l
    weighted_blos_r = num_r/denom_r
    return np.array((I, V, faraday, disp, unity_r, uidx_r, unity_l, uidx_l, weighted_blos_r, weighted_blos_l), dtype=np.float32)

@njit(parallel=True)
def apply_losint(ftemp, fbtot, fblos, fne, fdl, v, dogyro=False):
    """
    Parallelized line-of-sight synthesis for multiple rays.

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
    out : ndarray, shape (R, 10)
        Synthesized outputs for all rays:
        - Stokes I [K]
        - Stokes V [K]
        - Faraday rotation [rad]
        - Dispersion measure [pc cm^-3]
        - Left-handed optical unity location [cm]
        - Right-handed optical unity location [cm]
        - Left-handed optical unity index
        - Right-handed optical unity index
        - Left-handed emissivity-weighted BLOS
        - Right-handed emissivity-weighted BLOS
    """
    out = np.empty((ftemp.shape[0], 10), dtype=np.float32)
    for i in prange(ftemp.shape[0]):
        out[i] = losint(ftemp[i], fbtot[i], fblos[i], fne[i], fdl[i], v, dogyro)
    return out

@njit(parallel=True)
def range_losint(ftemp, fbtot, fblos, fne, fdl, vs, dogyro=False):
    """
    Parallelized line-of-sight synthesis for multiple frequencies.

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
    vs : ndarray, shape(F)
        Wave frequencies to synthesize [Hz].
    dogyro : bool, optional
        Whether to include gyroresonance absorption (default False).

    Returns
    -------
    out : ndarray, shape (F, R, 10)
        Synthesized outputs for all rays:
        - Stokes I [K]
        - Stokes V [K]
        - Faraday rotation [rad]
        - Dispersion measure [pc cm^-3]
        - Left-handed optical unity location [cm]
        - Right-handed optical unity location [cm]
        - Left-handed optical unity index
        - Right-handed optical unity index
        - Left-handed emissivity-weighted BLOS
        - Right-handed emissivity-weighted BLOS
    """
    raw_imgs = np.empty((vs.shape[0], ftemp.shape[0], 10), dtype=np.float32)
    for i in prange(vs.shape[0]):
        raw_imgs[i] = apply_losint(ftemp, fbtot, fblos, fne, fdl, vs[i], dogyro)
    return raw_imgs

kB = 1.38e-16 # erg/K
cc = 2.99792458e10 # cm/s

class Quantity(np.ndarray):
    """
    A physical quantity that wraps a NumPy array with associated units and an optional frequency.

    Attributes
    ----------
    unit : str
        The unit of the quantity (e.g., 'K', 'cm', 'Jy/beam', 'idx').
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

    def __new__(cls, arr, unit:str|None='cm', v:np.float32|None=None):
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
        if self.v is None and (unit == 'Jy/beam' or self.unit == 'Jy/beam'):
            raise ValueError("Frequency 'v' was not provided")
        return Quantity(self._UNITS[key](self), unit, self.v)
    
    def __repr__(self):
        base = f"Quantity({np.asarray(self)}, unit={self.unit!r})"
        return base
    
    def __str__(self):
        if self.shape == ():
            return f"{float(self):.4g}{self.unit}"
        else:
            return f"{np.array2string(self, precision=4, suppress_small=True)} {self.unit}"


class HandedQuantity(Quantity):
    def __new__(cls, left: Quantity, right: Quantity, both: Quantity | None = None):
        if both is None:
            avg_data = (left + right) / 2
            if np.issubdtype(left.dtype, np.integer):
                avg_data = np.asarray(np.round(avg_data).astype(left.dtype))
            both = Quantity(avg_data, unit=left.unit, v=left.v)
        obj = super().__new__(cls, both, both.unit, both.v)
        obj._left = left
        obj._right = right
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._left = getattr(obj, '_left', None)
        self._right = getattr(obj, '_right', None)
        self.unit = getattr(obj, 'unit', None)
        self.v = getattr(obj, 'v', None)

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right
    
    def to(self, unit):
        left_conv = self.left.to(unit)
        right_conv = self.right.to(unit)
        both_conv = (left_conv + right_conv) / 2
        return HandedQuantity(left_conv, right_conv, both_conv)

    def reshape(self, *shape, **kwargs):
        reshaped_both = super().reshape(*shape, **kwargs)
        left_reshaped = self.left.reshape(*shape, **kwargs)
        right_reshaped = self.right.reshape(*shape, **kwargs)
        return HandedQuantity(left_reshaped, right_reshaped, reshaped_both)

    def __repr__(self):
        return f"HandedQuantity({super().__repr__()}, left={self.left}, right={self.right})"

    def __getitem__(self, key):
        sliced_both = super().__getitem__(key)
        if not isinstance(sliced_both, Quantity):
            sliced_both = Quantity(sliced_both, unit=self.unit, v=self.v)
        sliced_left = self.left[key] if self.left is not None else None
        sliced_right = self.right[key] if self.right is not None else None
        return HandedQuantity(sliced_left, sliced_right, sliced_both)
    
    def __float__(self):
        return float(self.item())


class Image:
    """
    Represents a synthesized image from radiative transfer, containing Stokes I, V, and related physical maps.

    Attributes
    ----------
    I : Quantity
        Stokes I intensity image in Kelvin.
    V : Quantity
        Stokes V (circular polarization) image in Kelvin.
    P : Quantity or ndarray
        Fractional circular polarization, computed as V / I.
    faraday : Quantity
        Faraday rotation angle map in radians.
    disp : Quantity
        Dispersion measure map in units of pc cm^-3.
    unity : Quantity
        Optical unity position [cm], representing the average of left-handed and right-handed components. Can access the .left/.right attributes of uidx to isolate left-/right-handed components.
    uidx : Quantity
        Optical unity position indices, representing the average of left-handed and right-handed index components. Can access the .left/.right attributes of uidx to isolate left-/right-handed components.
    weighted_blos : Quantity
        Emissivity-weighted BLOS, representing the average of left-handed and right-handed index components. Can access the .left/.right attributes of weighted_blos to isolate left-/right-handed components.
    shape : tuple
        Shape of the underlying spatial data array (all but last dimension of input).
    v : float or None
        Frequency (in Hz) at which the image was generated or observed.
    """
    def __init__(self, pixels: NDArray[np.float32], v=None):
        self.I = Quantity(pixels[..., 0], 'K', v)
        self.V = Quantity(pixels[..., 1], 'K', v)
        self.P = self.V/self.I
        self.faraday = Quantity(pixels[..., 2], 'rad') 
        self.disp = Quantity(pixels[..., 3], 'pc cm^-3')
        self.unity = HandedQuantity(Quantity(pixels[..., 6], 'cm'), Quantity(pixels[..., 4], 'cm'))
        self.uidx = HandedQuantity(Quantity(pixels[..., 7].astype(np.int32), ''), Quantity(pixels[..., 5].astype(np.int32), ''))
        self.weighted_blos = HandedQuantity(Quantity(pixels[..., 9], 'G'), Quantity(pixels[..., 8], 'G'))
        self.shape = pixels.shape[:-1]
        self.v = v
    
    def reshape(self, *new_shape):
        self.I = self.I.reshape(new_shape)
        self.V = self.V.reshape(new_shape)
        self.P = self.P.reshape(new_shape)
        self.faraday = self.faraday.reshape(new_shape)
        self.disp = self.disp.reshape(new_shape)
        self.unity = self.unity.reshape(new_shape)
        self.uidx = self.uidx.reshape(new_shape)
        self.shape = new_shape
        return self
    
class ImageCollection(list):
    """
    A container class for managing multiple Image instances at different frequencies. Supports vectorized operations on images.
    """
    def __init__(self, images: list[Image] | None = None):
        super().__init__(images or [])

    def reshape(self, *new_shape):
        """
        Reshape all images in the collection to the given shape.
        """
        for image in self:
            image.reshape(*new_shape)
        return self
    
    def _stack_attr(self, attr: str):
        attrs = [getattr(img, attr) for img in self]
        if not attrs:
            raise ValueError(f"ImageCollection has no images to stack attribute '{attr}'.")
        first_attr = attrs[0]
        unit = getattr(first_attr, 'unit', None)
        v = getattr(first_attr, 'v', None)
        if getattr(first_attr, 'left', None) is not None:
            left_stack = np.stack([a.left for a in attrs])
            right_stack = np.stack([a.right for a in attrs])
            left_q = Quantity(left_stack, unit, v)
            right_q = Quantity(right_stack, unit, v)
            return HandedQuantity(left_q, right_q)
        else:
            stacked_values = np.stack([a.value if hasattr(a, 'value') else a for a in attrs])
            return Quantity(stacked_values, unit, v)
    
    @property
    def v(self) -> NDArray[np.float32]:
        """
        Return a list of frequency values (`v`) from all images in the collection.

        Returns
        -------
        ndarray
            The frequency `v` from each Image. If an Image has no `v`, None is returned for that Image.
        """
        return np.array([getattr(img, 'v', None) for img in self])

    @property
    def I(self):
        return self._stack_attr('I')

    @property
    def V(self):
        return self._stack_attr('V')
    
    @property
    def P(self):
        return self._stack_attr('P')

    @property
    def faraday(self):
        return self._stack_attr('faraday')

    @property
    def disp(self):
        return self._stack_attr('disp')

    @property
    def unity(self):
        return self._stack_attr('unity')

    @property
    def uidx(self):
        return self._stack_attr('uidx')
    
    @property
    def shape(self):
        if len(self) == 0:
            raise ValueError("ImageCollection is empty, shape is undefined.")
        return self[0].shape

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
        Flattened electron density array, shape (N, L).
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
            Electron density array.
        dl : float or ndarray
            Path length per voxel along the integration direction.
        axis : int, default=-1
            Axis along which to integrate (e.g., line-of-sight). A value of -1 (default) indicates last axis.
        """
        shape = temp.shape
        assert btot.shape == shape, "Quantities have mismatched shapes"
        assert blos.shape == shape, "Quantities have mismatched shapes"
        assert ne.shape == shape, "Quantities have mismatched shapes"
        if not isinstance(dl, np.ndarray):
            dl = np.broadcast_to(dl, shape)
        else:
            assert dl.shape == shape, "Quantities have mismatched shapes"

        with Wheel("creating rays...", label="quantities", total=5) as wheel:
            fray = lambda arr: self._unravel(np.ascontiguousarray(arr), axis)
            self.ftemp = fray(temp)
            wheel.update()
            self.fbtot = fray(btot)
            wheel.update()
            self.fblos = fray(blos)
            wheel.update()
            self.fne = fray(ne)
            wheel.update()
            self.fdl = fray(dl)
            wheel.update()
    
    def _unravel(self, arr, axis):
        axis = axis % arr.ndim
        arr_moved = np.moveaxis(arr, axis, -1)
        self.shape = arr_moved.shape # (nrays, ray_npt)
        leading = np.prod(self.shape[:-1], dtype=int)
        trailing = self.shape[-1]
        new_shape = (leading, trailing)
        return arr_moved.reshape(new_shape)

def synthesize(rays: RayCollection, v: np.float32 | float, dogyro:bool = False) -> Image:
    """
    Performs line-of-sight synthesis at a single frequency to compute a multi-channel image.

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
        A 2D multi-channel image object containing Stokes I, V, Faraday rotation, dispersion, and optical unity position/index.
    """
    with Wheel("synthesizing...") as wheel:
        fimg_raw = apply_losint(rays.ftemp, rays.fbtot, rays.fblos, rays.fne, rays.fdl, v, dogyro)
        rimg = fimg_raw.reshape(*rays.shape[:-1], 10)
    return Image(rimg, v)

def synthesize_range(rays: RayCollection, frequencies: list[float] | NDArray[np.float32], dogyro: bool = False) -> ImageCollection:
    """
    Performs line-of-sight synthesis for a range of frequencies.

    Parameters
    ----------
    rays : RayCollection
        Input ray collection with all physical parameters.
    frequencies : list | ndarray
        Frequencies to synthesize.
    dogyro : bool, default=False
        Whether to include gyroresonance effects in the calculation.

    Returns
    -------
    ImageCollection
        A collection of synthesized images.
    """
    results = ImageCollection()
    wheel = Wheel("synthesizing...", label="frequencies")
    for v in wheel(frequencies):
        fimg_raw = apply_losint(rays.ftemp, rays.fbtot, rays.fblos, rays.fne, rays.fdl, v, dogyro)
        rimg = fimg_raw.reshape(*rays.shape[:-1], 10)
        img = Image(rimg, v)
        results.append(img)
    return results

    # parallelized version
    # result = ImageCollection()
    # farr = np.array(frequencies, dtype=np.float32)
    # with Wheel("synthesizing...") as wheel:
    #     fimgs_raw = range_losint(rays.ftemp, rays.fbtot, rays.fblos, rays.fne, rays.fdl, farr, dogyro)
    #     rimg = fimgs_raw.reshape(farr.shape[0], *rays.shape[:-1], 10)
    # for i in range(farr.shape[0]):
    #     result.append(Image(rimg[i], farr[i]))
    # return result

def get_spectral_idx(images: ImageCollection) -> NDArray[np.float32]:
    """
    Compute the spectral index from an image collection. The spectral index is defined as n = -d(log10(Tb)) / d(log10(ν))

    Parameters
    ----------
    images : ImageCollection
        Collection of synthesized images.

    Returns
    -------
    spectral_idx : NDArray[np.float32]
        Array of spectral index values per frequency image per pixel (same shape as the input images).
    """
    vs_comp = images.v.reshape((-1,) + (1,) * len(images.shape))
    Tb = images.I
    dlogTb = np.gradient(np.log10(Tb), axis=0)
    dlogv = np.gradient(np.log10(vs_comp), axis=0)
    spectral_idx = -dlogTb/dlogv
    return spectral_idx

def invert_blos(images: ImageCollection) -> NDArray[np.float32]:
    """
    Estimate the line-of-sight magnetic field using circular polarization and spectral index.

    The inversion is based on:
        BLOS = P * ν / (2.8e6 * n)

    Parameters
    ----------
    images : ImageCollection
        Collection of Stokes `I` and `V` images at different frequencies.

    Returns
    -------
    inverted_blos : NDArray[np.float32]
        Estimated Bₗₒₛ values for each frequency and pixel in the image collection.
    """
    P = images.V/images.I
    ndim = images[0].V.ndim
    vs_comp = images.v.reshape((-1,) + (1,) * ndim)
    spectral_idx = get_spectral_idx(images)
    inverted_blos = ((P*vs_comp)/(2.8e6*spectral_idx)).astype(np.float32)
    return inverted_blos

def get_quantity_at_index(rays: RayCollection, quantity: str, indices: NDArray[np.int32]) -> Quantity:
    """
    Sample a given ray quantity at an array of indices.

    Parameters
    ----------
    rays : RayCollection
        Collection of ray-traced physical quantities (e.g., temperature, btot, etc.).
    indices : ndarray
        An array containing the indices of the ray quantities to sample.
    quantity : str
        Name of the quantity to sample (e.g., "temp", "blos", "btot").

    Returns
    -------
    Quantity
        The requested quantity sampled at the optical unity index along each ray.
    """
    cube = np.asarray(getattr(rays, 'f' + quantity))
    unit = {'blos': 'G', 'ne': 'cm^-3', 'temp': 'K', 'btot': 'G', 'dl': 'cm'}.get(quantity, '')
    indices = np.asarray(indices, dtype=np.int32)
    original_shape = indices.shape
    flat_indices = indices.ravel()
    if cube.shape[0] != flat_indices.size:
        raise ValueError(
            f"Mismatch: f{quantity} has {cube.shape[0]} rays, but got {flat_indices.size} indices"
        )
    sampled = cube[np.arange(flat_indices.size), flat_indices]
    sampled = sampled.reshape(original_shape)
    return Quantity(sampled, unit=unit)

def get_quantity_at_unity(rays: RayCollection, quantity: str, images: Image | ImageCollection):
    """
    Sample a given quantity at the left-/right-/average-handed unity layer for a single or collection of images.

    Parameters
    ----------
    rays : RayCollection
        Collection of ray-traced physical quantities.
    quantity : str
        Name of the quantity to sample (e.g., "temp", "blos", "btot").
    images : Image | ImageCollection
        A single Image or an ImageCollection containing `uidx` attributes.

    Returns
    -------
    Quantity
        Quantity sampled at optical unity with members `.left`, `.right` for handed components.
    """
    if isinstance(images, Image):
        images = ImageCollection([images])
    left_samples = [get_quantity_at_index(rays, quantity, img.uidx.left) for img in images]
    right_samples = [get_quantity_at_index(rays, quantity, img.uidx.right) for img in images]
    both_samples = [get_quantity_at_index(rays, quantity, img.uidx) for img in images]
    unit = {'blos': 'G', 'ne': 'cm^-3', 'temp': 'K', 'btot': 'G', 'dl': 'cm'}.get(quantity, '')
    left_qty = Quantity(np.stack(left_samples), unit=unit)
    right_qty = Quantity(np.stack(right_samples), unit=unit)
    both_qty = Quantity(np.stack(both_samples), unit=unit)
    return HandedQuantity(left_qty, right_qty, both=both_qty)

def get_emissivity_weighted_blos(rays: RayCollection) -> Quantity:
    """
    Compute the emissivity-weighted average LOS magnetic field along each ray.
    """
    blos = rays.fblos     # shape: (n_rays, n_steps)
    eta = rays.feta_I     # same shape
    tau = rays.ftau       # optical depth to observer
    dl = rays.fdl         # cm

    # Compute the contribution function
    contrib = eta * np.exp(-tau) * dl

    # Numerator: ∫ B_LOS * contrib
    weighted_blos = np.sum(blos * contrib, axis=1)

    # Denominator: ∫ contrib
    total_contrib = np.sum(contrib, axis=1)

    avg_blos = weighted_blos / total_contrib
    return Quantity(avg_blos, unit='G')