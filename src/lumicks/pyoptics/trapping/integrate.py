import numpy as np

SQ = 5**0.5
v1 = (1, 0, 0)
v2 = (1 / SQ, 2 / SQ, 0)
v3 = (1 / SQ, (1 - SQ**-1) / 2, ((1 + SQ**-1) / 2) ** 0.5)
v4 = (1 / SQ, (1 - SQ**-1) / 2, -(((1 + SQ**-1) / 2) ** 0.5))
v5 = (1 / SQ, (-1 - SQ**-1) / 2, ((1 - SQ**-1) / 2) ** 0.5)
v6 = (1 / SQ, (-1 - SQ**-1) / 2, -(((1 - SQ**-1) / 2) ** 0.5))
v7 = (-1, 0, 0)
v8 = (-1 / SQ, -2 / SQ, 0)
v9 = (-1 / SQ, -(1 - SQ**-1) / 2, -(((1 + SQ**-1) / 2) ** 0.5))
v10 = (-1 / SQ, -(1 - SQ**-1) / 2, (((1 + SQ**-1) / 2) ** 0.5))
v11 = (-1 / SQ, (1 + SQ**-1) / 2, -(((1 - SQ**-1) / 2) ** 0.5))
v12 = (-1 / SQ, (1 + SQ**-1) / 2, (((1 - SQ**-1) / 2) ** 0.5))


icosahedron_faces = np.asarray(
    [
        [v1, v2, v3],
        [v1, v3, v5],
        [v1, v5, v6],
        [v1, v6, v4],
        [v1, v4, v2],
        [v7, v12, v11],
        [v7, v11, v9],
        [v7, v9, v8],
        [v7, v8, v10],
        [v7, v10, v12],
        [v2, v11, v12],
        [v2, v12, v3],
        [v3, v12, v10],
        [v3, v10, v5],
        [v5, v10, v8],
        [v5, v8, v6],
        [v6, v8, v9],
        [v4, v6, v9],
        [v4, v9, v11],
        [v2, v4, v11],
    ]
)


def subdivide_faces(faces):
    shape = faces.shape
    new_faces = np.empty((4 * shape[0], *shape[1:]))
    for idx, face in enumerate(faces):
        e1 = (face[0] + face[1]) * 0.5
        e2 = (face[0] + face[2]) * 0.5
        e3 = (face[1] + face[2]) * 0.5

        new_faces[idx * 4, :, :] = [face[0], e1, e2]
        new_faces[idx * 4 + 1, :, :] = [e1, face[1], e3]
        new_faces[idx * 4 + 2, :, :] = [e2, e3, face[2]]
        new_faces[idx * 4 + 3, :, :] = [e1, e3, e2]
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
    w = [1 / (20 * 4 ** (factor - 1))] * len(x)
    return x, y, z, w
