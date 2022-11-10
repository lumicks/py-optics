import numpy as np


# Some constants useful for calculating the magnetic fields
_C = 299792458
_MU0 = 4 * np.pi * 1e-7
_EPS0 = (_C**2 * _MU0)**-1

def field_dipole_x(px, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of an x-oriented dipole. The field includes
    both near- and farfields. The dipole is located at (0,0,0).
    Reference: Principles of Nano-optics, 2nd Ed., Ch. 2 and "a little algebra"
    
    Arguments:
        px: dipole moment of the dipole (SI units)
        n_medium: refractive index of the medium in which the dipole is embedded
        lambda_vac: wavelength in vacuum of the radiation
        x, y, z: (array of) coordinates at which the electromagnetic field is to
            be evaluated
        
        Returns:
            Ex: array of electric field polarized in the x-direction 
                evaluated at (x, y, z)
            Ey: As Ex, but y-polarized component
            Ez: As Ex, but z-polarized component
            Hx: H field polarized in the x-direction evaluated at (x, y, z)
            Hy: As Hx, but y-polarized component
            Hz: As Hx, but z-polarized component
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    k = 2 * np.pi * n_medium/lambda_vac
    R = np.hypot(np.hypot(x, y), z)

    fix = (R == 0)
    R[fix] = 1
    
    Ex = 1 + 1 / (k**2 * R) * (-k**2 * x**2 / R + 
        (2 * x**2 - y**2 - z**2) * (1 / R**3 - 1j * k / R**2))
    Ey = x * y / (k**2 * R) * (3 / R**3 - 3j * k / R**2 - k**2 / R)
    Ez = x * z / (k**2 * R) * (3 / R**3 - 3j * k / R**2 - k**2 / R)

    prefactor = px * k**2 * np.exp(1j * k * R)/(4 * np.pi * R * _EPS0 * n_medium**2)
    Ex *= prefactor
    Ey *= prefactor
    Ez *= prefactor

    Hy = 1j * k * z / R - z / R**2
    Hz = y / R**2 - 1j * k * y / R

    prefactor *= (1j * 2 * np.pi * _C / lambda_vac * _MU0)**-1

    Hy *= prefactor
    Hz *= prefactor

    Hx = np.zeros(Hy.shape, dtype='complex128')

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j*np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz


def field_dipole_y(py, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of a y-oriented dipole. The field includes
    both near- and farfields. The implementation is based on a permutation of
    the fields as calculated by fields_dipole_z.
    
    Arguments:
        py: dipole moment of the dipole (SI units)
        n_medium: refractive index of the medium in which the dipole is embedded
        lambda_vac: wavelength in vacuum of the radiation
        x, y, z: (array of) coordinates at which the electromagnetic field is to
            be evaluated
        
        Returns:
            Ex: array of electric field polarized in the x-direction 
                evaluated at (x, y, z)
            Ey: As Ex, but y-polarized component
            Ez: As Ex, but z-polarized component
            Hx: H field polarized in the x-direction evaluated at (x, y, z)
            Hy: As Hx, but y-polarized component
            Hz: As Hx, but z-polarized component
    """

    Ex, Ez, Ey, Hx, Hz, Hy = field_dipole_z(py, n_medium, lambda_vac, x, -z, y)
    Ez *= -1
    Hz *= -1
    return Ex, Ey, Ez, Hx, Hy, Hz

def field_dipole_z(pz, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic field of a z-oriented dipole. The field includes
    both near- and farfields. The dipole is located at (0,0,0).
    Reference: Antenna Theory, Ch. 4, 3rd Edition, C. A. Balanis
    
    Arguments:
        pz: dipole moment of the dipole (SI units)
        n_medium: refractive index of the medium in which the dipole is embedded
        lambda_vac: wavelength in vacuum of the radiation
        x, y, z: (array of) coordinates at which the electromagnetic field is to
            be evaluated
        
        Returns:
            Ex: array of electric field polarized in the x-direction 
                evaluated at (x, y, z)
            Ey: As Ex, but y-polarized component
            Ez: As Ex, but z-polarized component
            Hx: H field polarized in the x-direction evaluated at (x, y, z)
            Hy: As Hx, but y-polarized component
            Hz: As Hx, but z-polarized component
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    rho = np.hypot(x,y)
    r = np.hypot(rho,z)

    fix = r == 0
    r[fix] = 1
    cosT = z / r
    sinT = (1 - cosT**2)**0.5
    
    # Division by zero when x == y == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        cosP = x / rho
        sinP = y / rho
    
    # based on physics/symmetry, the field is purely z-oriented whenever x == y
    # == 0. Therefore we can safely set the cosP and sinP to zero in that case
    # as it does not affect Ez (no cosP or sinP terms)
    cosP[rho == 0] = sinP[rho == 0] = 0

    _eps = _EPS0 * n_medium**2
    eta = (_MU0 / _eps)**0.5
    k = 2 * np.pi * n_medium / lambda_vac
    w = _C * (k / n_medium)

    I0l = 1j * w * pz

    Er =  eta * I0l * cosT / (2 * np.pi * r**2) * (1 + 
        1 / (1j * k * r)) * np.exp(-1j * k * r)
    Et = 1j * eta * k *I0l *sinT/(4*np.pi*r)*(1 + (1j*k*r)**-1 - (k*r)**-2) * np.exp(-1j*k*r)

    Hp = 1j*k*I0l*sinT/(4*np.pi*r)*(1+(1j*k*r)**-1)*np.exp(-1j*k*r)

    Ex = Er * sinT * cosP + Et * cosT * cosP
    Ey = Er * sinT * sinP + Et * cosT * sinP
    Ez = Er * cosT - Et * sinT

    Hx = - Hp * sinP
    Hy = Hp * cosP
    Hz = np.zeros(Hp.shape, dtype='complex128')

    # Balanis uses exp(1j * w * t) as a convention, we use exp(-1j * w * t)
    Ex = np.conj(Ex)
    Ey = np.conj(Ey)
    Ez = np.conj(Ez)

    Hx = np.conj(Hx)
    Hy = np.conj(Hy)

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j*np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz


def field_dipole(p, n_medium, lambda_vac, x, y, z, farfield=False):
    """Get the electromagnetic field of an arbitrarily-oriented dipole. 
    The field includes both near- and farfields. The dipole is located at (0,0,0)
    Reference: Classical Electrodynamics, Ch. 9, 3rd Edition, J.D. Jackson
        
        Arguments:
            p: tuple of (px, py, pz): the dipole moment of the dipole (SI units)
            n_medium: refractive index of the medium in which the dipole is embedded
            lambda_vac: wavelength in vacuum of the radiation
            x, y, z: (array of) coordinates at which the electromagnetic field is to
                be evaluated
            
            Returns:
                Ex: array of electric field polarized in the x-direction 
                    evaluated at (x, y, z)
                Ey: As Ex, but y-polarized component
                Ez: As Ex, but z-polarized component
                Hx: H field polarized in the x-direction evaluated at (x, y, z)
                Hy: As Hx, but y-polarized component
                Hz: As Hx, but z-polarized component
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    r = np.hypot(np.hypot(x,y),z)
    k = 2 * np.pi * n_medium / lambda_vac
    fix = r == 0
    r[fix] = 1
    nx = x/r
    ny = y/r
    nz = z/r
    px = p[0]
    py = p[1]
    pz = p[2]

    nxp_x = ny*pz - nz*py
    nxp_y = nz*px - nx*pz
    nxp_z = nx*py - ny*px

    nxpxn_x = nxp_y*nz-nxp_z*ny
    nxpxn_y = nxp_z*nx-nxp_x*nz
    nxpxn_z = nxp_x*ny-nxp_y*nx

    ndotp = nx*px + ny*py + nz*pz

    eps_inv = (_EPS0 * n_medium**2)**-1
    G0 = np.exp(1j * k * r) / (4 * np.pi * r)

    if farfield:
        Ex = k**2 * nxpxn_x + 0j
        Ey = k**2 * nxpxn_y + 0j
        Ez = k**2 * nxpxn_z + 0j
        Hx = np.ones(Ex.shape, dtype='complex128')
    else:
        Ex = k**2 * nxpxn_x + (3 * nx * ndotp - px) * (r**-2 - 1j * k / r)
        Ey = k**2 * nxpxn_y + (3 * ny * ndotp - py) * (r**-2 - 1j * k / r)
        Ez = k**2 * nxpxn_z + (3 * nz * ndotp - pz) * (r**-2 - 1j * k / r)
        Hx = 1 - (1j * k * r)**-1
    Hy = Hx.copy()
    Hz = Hx.copy()
    
    Ex *= G0 * eps_inv
    Ey *= G0 * eps_inv
    Ez *= G0 * eps_inv
    
    prefactor_H = G0 * _C * k**2 / n_medium

    Hx *= prefactor_H * nxp_x
    Hy *= prefactor_H * nxp_y
    Hz *= prefactor_H * nxp_z

    Ex[fix] = Ey[fix] = Ez[fix] = Hx[fix] = Hy[fix] = Hz[fix] = np.nan + 1j*np.nan

    return Ex, Ey, Ez, Hx, Hy, Hz

def farfield_dipole_position(p, n_medium, lambda_vac, x, y, z):
    """Get the electromagnetic farfield of an arbitrarily-oriented dipole. 
    The dipole is located at (0,0,0).
    Reference: Principles of Nano-optics, 2nd Ed., Appendix D
        
        Arguments:
            p: tuple of (px, py, pz): the dipole moment of the dipole (SI units)
            n_medium: refractive index of the medium in which the dipole is embedded
            lambda_vac: wavelength in vacuum of the radiation
            x, y, z: Array of locations in the far field where to evaluate the electric
            field (meters). 
            
            Returns:
                Ex: array of electric field polarized in the x-direction 
                    evaluated at (x, y, z)
                Ey: As Ex, but y-polarized component
                Ez: As Ex, but z-polarized component
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    assert x.shape == y.shape == z.shape, \
        'Location parameters x, y and z need to be of the same shape'
    r = np.hypot(x,np.hypot(y,z))
    Sx = x / r
    Sy = y / r
    Sz = z / r
    
    k = 2 * np.pi * n_medium / lambda_vac
    prefactor = k**2 * np.exp(1j * k * r) / (n_medium**2 * _EPS0 * 
        4 * np.pi * r)
    Ex = (p[0] * (1 - Sx**2) - p[1] * Sx * Sy 
        - p[2] * Sx * Sz) * prefactor
    Ey = (-p[0] * Sx * Sy + p[1] * (1 - Sy**2)
        - p[2] * Sy * Sz) * prefactor
    Ez = (-p[0] * Sx * Sz - p[1] * Sy * Sz
        + p[2] * (1 - Sz**2)) * prefactor

    return  Ex, Ey, Ez


def farfield_dipole_angle(p, n_medium, lambda_vac, cosPhi, sinPhi, cosTheta, r):
    """Get the electromagnetic farfield of an arbitrarily-oriented dipole. 
    The dipole is located at (0,0,0).
    Reference: Principles of Nano-optics, 2nd Ed., Appendix D
        
        Arguments:
            p: tuple of (px, py, pz): the dipole moment of the dipole (SI units)
            n_medium: refractive index of the medium in which the dipole is embedded
            lambda_vac: wavelength in vacuum of the radiation (meters)
            cosPhi: cosine of the angle Phi, which is the angle between the
            location (x, y, 0) and the x-axis.
            sinPhi: sine of the angle Phi
            cosTheta: cosine of the angle Theta, which is the angle of the
                location (x, y, z) with the z-axis.
            r: distance from (0, 0, 0) to (x, y, z)
            
            Returns:
                Ex: array of electric field polarized in the x-direction 
                    evaluated at (x, y, z)
                Ey: As Ex, but y-polarized component
                Ez: As Ex, but z-polarized component
    """
    cosPhi = np.atleast_1d(cosPhi)
    sinPhi = np.atleast_1d(sinPhi)
    cosTheta = np.atleast_1d(cosTheta)

    assert cosPhi.shape == sinPhi.shape == cosTheta.shape

    sinT = np.zeros(cosTheta.shape)
    sinT[cosTheta <= 1] = (1 - cosTheta[cosTheta <= 1]**2)**0.5
    
    k = 2 * np.pi * n_medium / lambda_vac
    prefactor = k**2 * np.exp(1j * k * r) / (n_medium**2 * _EPS0 * 
        4 * np.pi * r)
    Ex = (p[0] * (1 - cosPhi**2 * sinT**2) - p[1] * sinPhi * cosPhi * sinT**2 
        - p[2] * cosPhi * sinT * cosTheta) * prefactor
    Ey = (-p[0] * sinPhi * cosPhi * sinT**2 + p[1] * (1 - sinPhi**2 * sinT**2)
        - p[2] * sinPhi * sinT * cosTheta) * prefactor
    Ez = (-p[0] * cosPhi * sinT * cosTheta - p[1] * sinPhi * sinT * cosTheta
        + p[2] * sinT**2) * prefactor

    return  Ex, Ey, Ez
