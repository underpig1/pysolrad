import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from collections import namedtuple

@njit
def gyrotaus(v, s, ne, temp, theta_rad, dl):
    # params: wave freq, harmonic n, ion density, temp, angle between B and k, line of sight component
    # return: x and o mode components of optical depth

    if s < 1 or s > 8:
        return 0.0, 0.0
    me = 9.11e-28 # g
    kB = 1.38e-16 # erg/K
    cc = 3.0e10 # cm/s
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
    f = 2.8e6 * blos/v
    f = max(min(f, 0.99), 0.01)
    dtau = 0.2 * ne**2/v**2*Te**(-3./2.)*dl
    dtaur, dtaul = dtau/(1.-f)**2., dtau/(1.+f)**2.
    if btot > 0 and dogyro:
        s_eff = v / (2.8e6 * btot)
        s = int(np.round(s_eff))
        if 1 <= s <= 8 and np.abs(s - s_eff) < 0.2:
            theta_rad = np.arccos(min(max(blos / btot, -1.0), 1.0))
            taux, tauo = _gyrotaus(v, s, ne, Te, theta_rad, dl)
            if blos < 0:
                taux, tauo = tauo, taux
            dtaur += taux
            dtaul += tauo
    return dtaur, dtaul

@njit
def radtrans(dtaur, dtaul, Te, Tbr, Tbl):
    Tbr, Tbl = Tbr*np.exp(-dtaur)+Te*(1.-np.exp(-dtaur)), Tbl*np.exp(-dtaul)+Te*(1.-np.exp(-dtaul))
    return Tbr, Tbl

@njit
def losint(ltemp, lbtot, lblos, lne, ldl, v, dogyro=False):
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
        dtaur, dtaul = taus(pblos, pbtot, pne, ptemp, pdl, v, dogyro)
        integrands[:, il] = dtaur, dtaul
        Tbr, Tbl = radtrans(dtaur, dtaul, ptemp, Tbr, Tbl)
        rm += pne * pblos * pdl
        disp += pne*pdl
    I = (Tbr + Tbl) / 2.
    V = (Tbr - Tbl) / 2.
    faraday = 2.63e-13 * rm * (3e8 / v)**2
    unity = 0.0
    for il in range(npts - 1, -1, -1):
        dtaur, dtaul = integrands[:, il]
        taul += dtaul
        taur += dtaur
        unity += ldl[il]
        if (taul + taur)/2 >= 1.0:
            break
    return np.array((I, V, faraday, disp, unity), dtype=pixel)


# class Pixel:
#     def __init__(self, I: np.float32, V: np.float32, faraday: np.float32, disp: np.float32, unity: np.float32):
#         self.I = I
#         self.V = V
#         self.faraday = faraday
#         self.disp = disp
#         self.unity = unity

# Pixel = namedtuple('Pixel', ['I', 'V', 'faraday', 'disp', 'unity'])
pixel = np.dtype([('I', np.float32), ('V', np.float32), ('faraday', np.float32), ('disp', np.float32), ('unity', np.float32)])

class LineOfSight:
    def __init__(self, ltemp: NDArray[np.float32], lbtot: NDArray[np.float32], lblos: NDArray[np.float32], lne: NDArray[np.float32], ldl: NDArray[np.float32]):
        self.ltemp = ltemp
        self.lbtot = lbtot
        self.lblos = lblos
        self.lne = lne
        self.ldl = ldl

    def integrate(self, v: np.float32, dogyro:bool = False) -> NDArray[np.void]:
        I, V, faraday, disp, unity = _losint(self.ltemp, self.lbtot, self.lblos, self.lne, self.ldl, v, dogyro)
        return np.array((I, V, faraday, disp, unity), dtype=pixel)

class DataCube:
    """
    A cube consisting of x, y, and z distributions of a quantity
    """
    def __init__(self, data):
        self.data = data

class VectorCube:
    """
    A cube consisting of
    """
    def __init__(self, xcube: DataCube, ycube: DataCube, zcube: DataCube):
        self.xcube = xcube
        self.ycube = ycube
        self.zcube = zcube

class Image:
    def __init__(self, pixels: NDArray[np.void]):
        self.I = pixels['I']
        self.V = pixels['V']
        self.faraday = pixels['faraday']
        self.disp = pixels['disp']
        self.unity = pixels['unity']
        self.shape = pixels.shape

class Atmosphere:
    def __init__(self, atemp: NDArray[np.float32], abtot: NDArray[np.float32], ablos: NDArray[np.float32], ane: NDArray[np.float32], adl: NDArray[np.float32] | np.float32, axis:int = 0):
        self.atemp = atemp
        self.abtot = abtot
        self.ablos = ablos
        self.ane = ane
        self.adl = adl

    def rays(self):
        return RayCollection()

rtemp: (x, y, z (ray_npt)) | (x (ray_npt)) -> LOS is last axis
class RayCollection:
    def __init__(self, rtemp: NDArray[np.float32], rbtot: NDArray[np.float32], rblos: NDArray[np.float32], rne: NDArray[np.float32], rdl: NDArray[np.float32]):
        self.rtemp = rtemp
        self.rbtot = rbtot
        self.rblos = rblos
        self.rne = rne
        self.rdl = rdl

        self.shape = rtemp.shape
        fray = lambda rq: rq.reshape(-1, self.shape[-1])
        self.ftemp = fray(rtemp)
        self.fbtot = fray(rbtot)
        self.fblos = fray(rblos)
        self.fne = fray(rne)
        self.fdl = fray(rdl)

@njit(parallel=True)
def apply_losint(ftemp, fbtot, fblos, fne, fdl, v, dogyro=False):
    out = np.empty(ftemp.shape[0], dtype=pixel)
    for i in prange(ftemp.shape[0]):
        out[i] = losint(ftemp[i], fbtot[i], fblos[i], fne[i], fdl[i], v, dogyro)
    return out

def synthesize(rays: RayCollection, v: np.float32, dogyro:bool = False) -> Image:
    fimg = apply_losint(rays.ftemp, rays.fbtot, rays.fblos, rays.fne, rays.fdl, v, dogyro)
    rimg = fimg.reshape(rays.shape)
    return Image(rimg)

img = synthesize(RayCollection atm)
plt.imshow(img.I)

@njit(parallel=True)
def genimg(temp, btot, bz, ne, dl, v, dogyro=False):
    shp = temp.shape
    pdl = dl
    FR = np.zeros((2, shp[0], shp[1]))
    IV = np.zeros((2, shp[0], shp[1])) # stokes IV
    THK = np.zeros((shp[0], shp[1])) # optical thickness
    INT = np.zeros((shp[0], shp[1], shp[2], 2)) # integrands
    for ix in prange(shp[0]):
        for iy in prange(shp[1]):
            I, V, FR, DM, UNITY = losint
    return IV, THK