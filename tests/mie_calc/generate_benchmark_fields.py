import numpy as np
import miepy

def create_ref_data(n_bead=1.5, n_medium=1.33, bead_diam=1e-6,
    lambda_vac=1064e-9, num_pts=100, filename = 'data'):

    bead = miepy.constant_material(n_bead**2, 1.0)
    medium = miepy.constant_material(n_medium**2, 1.0)

    size_param = 2*np.pi*bead_diam*n_medium/(2*lambda_vac)
    lmax = int(np.round(size_param + 4 * size_param**(1/3) + 2.0))
    sphere = miepy.single_mie_sphere(bead_diam/2, bead, lambda_vac, lmax, medium=medium)

    x = np.linspace(-bead_diam,bead_diam,100)
    y = x
    z = x

    X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
    R = np.hypot(np.hypot(X,Y),Z)
    Th = np.arccos(Z/R)
    Ph = np.arctan2(Y,X)

    E_func = sphere.E_field(index=0)
    E = E_func(R,Th,Ph).squeeze()
    Th = np.squeeze(Th)
    Ph = np.squeeze(Ph)

    sinT = np.sin(Th)
    cos_theta = np.cos(Th)
    sinP = np.sin(Ph)
    cosP = np.cos(Ph)

    Er = E[0,:,:]
    Et = E[1,:,:]
    Ep = E[2,:,:]
    Ex = Er * sinT * cosP + Et * cos_theta * cosP - Ep * sinP
    Ey = Er * sinT * sinP + Et * cos_theta * sinP + Ep * cosP
    Ez = Er * cos_theta - Et * sinT

    Ex = np.squeeze(Ex)
    Ey = np.squeeze(Ey)
    Ez = np.squeeze(Ez)

    np.savez_compressed(filename, x=x, y=y, z=z, Ex=Ex, Ey=Ey, Ez=Ez, 
        num_pts=num_pts, num_orders=lmax, n_bead=n_bead, n_medium = n_medium,
        bead_diam=bead_diam, lambda_vac=lambda_vac)

create_ref_data(filename='ref1')
create_ref_data(bead_diam=4.4e-6, filename='ref2')
create_ref_data(n_bead=0.1+2j, bead_diam=0.2e-6, n_medium=1.33, filename='ref3')
create_ref_data(n_bead=1.0, bead_diam=2e-6, filename='ref4')
create_ref_data(n_bead=1.33+1e-3j, bead_diam=0.8e-6, filename='ref5')