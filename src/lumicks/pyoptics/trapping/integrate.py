import numpy as np
from scipy.constants import golden_ratio as phi

icosahedron_faces = np.asarray(
    [
    [(0, -1, phi), (phi, 0, -1), (-phi, 0, -1)],
    [(0, -1, -phi), (-phi, 0, -1), (phi, 0, -1)],
    [(0, -1, phi), (-phi, 0, -1), (-1, -phi, 0)],
    [(0, -1, -phi), (-1, -phi, 0), (-phi, 0, -1)],
    [(0, -1, phi), (-1, -phi, 0), (1, -phi, 0)],
    [(0, -1, -phi), (1, -phi, 0), (-1, -phi, 0)],
    [(0, -1, phi), (1, -phi, 0), (phi, 0, -1)],
    [(0, -1, -phi), (phi, 0, -1), (1, -phi, 0)],
    [(0, 1, phi), (-phi, 0, 1), (phi, 0, 1)],
    [(0, 1, -phi), (phi, 0, 1), (-phi, 0, 1)],
    [(0, 1, phi), (phi, 0, 1), (1, phi, 0)],
    [(0, 1, -phi), (1, phi, 0), (phi, 0, 1)],
    [(0, 1, phi), (1, phi, 0), (-1, phi, 0)],
    [(0, 1, -phi), (-1, phi, 0), (1, phi, 0)],
    [(0, 1, phi), (-1, phi, 0), (-phi, 0, 1)],
    [(0, 1, -phi), (-phi, 0, 1), (-1, phi, 0)],
    [(0, -1, phi), (phi, 0, -1), (-phi, 0, 1)],
    [(0, -1, -phi), (-phi, 0, 1), (phi, 0, -1)],
    [(0, -1, phi), (-phi, 0, 1), (-1, -phi, 0)],
    [(0, -1, -phi), (-1, -phi, 0), (-phi, 0, 1)]
])

def subdivide_faces(faces):
    new_faces = []
    for face in faces:
        e1 = (face[0] + face[1]) * 0.5
        e2 = (face[1] + face[2]) * 0.5
        e3 = (face[2] + face[0]) * 0.5

        f1 = [face[0], e1, e3]
        f2 = [e1, face[1], e2]
        f3 = [e2, face[2], e3]
        f4 = [e1, e2, e3]
        new_faces.extend([f1, f2, f3, f4])
    return new_faces


def get_integration_locations(factor: int = 1):
    if factor < 1:
        raise ValueError("Factor cannot be smaller than 1")
    x = []
    y = []
    z = []
    faces = icosahedron_faces
    for _ in range(factor - 1):
        faces = subdivide_faces(faces)
    for face in faces:
        centroid = face[0] + face[1] + face[2]
        centroid /= np.linalg.norm(centroid)
        x.append(centroid[0])
        y.append(centroid[1])
        z.append(centroid[2])
    w = [1 / (20 * 4**(factor - 1))] * len(x)
    return x, y, z, w
