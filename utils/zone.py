import numpy as np
import logging
import magpylib as magpy
logger = logging.getLogger(__name__)

class Zone:
    def __init__(self, resolution=2e-3):
        logger.debug(f"Creating {type(self).__name__} with resolution {resolution}")
        self.coords = np.array([])
        self.space = np.array([])
        self.mask = np.array([])
        self.resolution = resolution
        self.zonelist = []
        self.zonelist.append(self)

    def constraints(self, space):
        return np.zeros_like(space[:, 0], dtype=bool)

    def update(self):
        if len(self.zonelist) > 1:
            [z.update() for z in self.zonelist if z is not self]
        zonelist = self.zonelist
        zonelist = [z for z in zonelist if not z.coords.size == 0]
        if len(zonelist) == 0:
            return
        elif len(zonelist) == 1:
            all_coords = zonelist[0].coords
        else:
            all_coords = np.vstack([z.coords for z in zonelist])
        self.coords = all_coords
        min_coords = np.min(self.coords, axis=0)
        max_coords = np.max(self.coords, axis=0)
        #sample = [np.linspace(min_coords[i], max_coords[i], int((max_coords[i] - min_coords[i]) / self.resolution) + 1) for i in range(3)]
        #self.space = np.array(np.meshgrid(sample[0], sample[1], sample[2])).T
        nx = int((max_coords[0] - min_coords[0]) / self.resolution) + 1
        ny = int((max_coords[1] - min_coords[1]) / self.resolution) + 1
        nz = int((max_coords[2] - min_coords[2]) / self.resolution) + 1
        self.space = np.mgrid[
            min_coords[0]:max_coords[0]:complex(0, nx),
            min_coords[1]:max_coords[1]:complex(0, ny),
            min_coords[2]:max_coords[2]:complex(0, nz)
        ].transpose(1,2,3,0)
        masks = [z.constraints(self.space.reshape(-1, 3)) for z in self.zonelist]
        self.mask = np.any(masks, axis=0).reshape(self.space.shape[:3])
        self.coords = self.space.reshape(-1, 3)[self.mask.flatten()]

    def isin(self, point):
        if not isinstance(point, np.ndarray):
            point = np.array(point, dtype=self.coords.dtype)
        if point.ndim == 1:
            rst = np.all(self.coords == point, axis=1)
            return np.any(rst), np.where(rst)[0][0]
        else:
            raise ValueError("Point must be a 1D or 2D array with shape (3,) or (N, 3).")

    def reshape(self, arr, fill=np.nan):
        if arr.shape[0] != len(self.coords):
            raise ValueError("Array length does not match the number of coordinates in the zone.")
        if len(arr.shape) > 1:
            rst = np.full(self.mask.shape + (arr.shape[-1],), fill_value=fill, dtype=arr.dtype)
        else:
            rst = np.full(self.mask.shape, fill_value=fill, dtype=arr.dtype)
        rst[self.mask] = arr
        return rst

    def getSensor(self):
        return magpy.Sensor(pixel=self.coords, style_size=2.5, style_opacity = 0.5)

    def genMeshPoint(self, profile):
        pointStr = ''
        pointStr += 'Print[ bz, OnPoint {'+str(self.coords[0, 0])+', '+str(self.coords[0, 1])+', '+str(self.coords[0, 2])+'}, Format Table, File "dsv.txt"];\n'
        for coord in self.coords[1:]:
            pointStr += 'Print[ bz, OnPoint {'+str(coord[0])+', '+str(coord[1])+', '+str(coord[2])+'}, Format Table, File >> "dsv.txt"];\n'
        profile = profile.replace('{DSV_TEMPLATE}', pointStr)
        return profile

    def genGrad(self, strength=0.015, gradType='z', axis=2):
        field = np.zeros_like(self.space, dtype=np.float32)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                for k in range(self.mask.shape[2]):
                    if self.mask[i,j,k] != 0:
                        x = self.space[i,j,k,0]
                        y = self.space[i,j,k,1]
                        z = self.space[i,j,k,2]
                        if gradType == 'z':
                            field[i,j,k,axis] = strength * z
                        elif gradType == 'x':
                            field[i,j,k,axis] = strength * x
                        elif gradType == 'y':
                            field[i,j,k,axis] = strength * y
        field = field[self.mask != 0,:]
        return field

    def __add__(self, other):
        if isinstance(other, Zone):
            logger.debug(f"Combining {type(self).__name__} with {type(other).__name__}")
            new_zone = Zone(self.resolution)
            new_zone.zonelist = self.zonelist + other.zonelist
            new_zone.update()
            return new_zone
        else:
            raise TypeError("Can only add another Zone instance.")

class SphereZone(Zone):
    def __init__(self, resolution=2e-3, center=(0, 0, 0), radius=0.01):
        super().__init__(resolution)
        self.radius = radius
        self.center = np.array(center, dtype=np.float32)
        sample = np.linspace(-radius, radius, int(radius / self.resolution) * 2 + 1)
        self.coords = np.array(np.meshgrid(sample, sample, sample)).T.reshape(-1, 3)
        self.coords = self.coords[np.linalg.norm(self.coords, axis=1) <= radius]
        self.coords += self.center
        self.update()

    def constraints(self, space):
        distances = np.linalg.norm(space - self.center, axis=1)
        return distances <= self.radius

class CubeZone(Zone):
    def __init__(self, resolution=2e-3, center=(0, 0, 0), size=0.01):
        super().__init__(resolution)
        self.size = size
        self.center = np.array(center, dtype=np.float32)
        half_size = size / 2
        sample = np.linspace(-half_size, half_size, int(size / self.resolution) + 1)
        self.coords = np.array(np.meshgrid(sample, sample, sample)).T.reshape(-1, 3)
        self.coords += self.center
        self.update()

    def constraints(self, space):
        return np.all(np.abs(space - self.center) <= (self.size / 2), axis=1)

class RectZone(Zone):
    def __init__(self, resolution=2e-3, center=(0, 0, 0), size=(0.01, 0.01, 0.01)):
        super().__init__(resolution)
        self.size = np.array(size, dtype=np.float32)
        self.center = np.array(center, dtype=np.float32)
        half_size = self.size / 2
        sample_x = np.linspace(-half_size[0], half_size[0], int(self.size[0] / self.resolution) + 1)
        sample_y = np.linspace(-half_size[1], half_size[1], int(self.size[1] / self.resolution) + 1)
        sample_z = np.linspace(-half_size[2], half_size[2], int(self.size[2] / self.resolution) + 1)
        self.coords = np.array(np.meshgrid(sample_x, sample_y, sample_z)).T.reshape(-1, 3)
        self.coords += self.center
        self.update()

    def constraints(self, space):
        return np.all(np.abs(space - self.center) <= (self.size / 2), axis=1)

class CylinderZone(Zone):
    def __init__(self, resolution=2e-3, center=(0, 0, 0), radius=0.01, height=0.01, axis='z'):
        super().__init__(resolution)
        self.radius = radius
        self.height = height
        self.axis = axis
        self.center = np.array(center, dtype=np.float32)
        sample = np.linspace(-radius, radius, int(radius / self.resolution) * 2 + 1)
        direction_sample = np.linspace(-height / 2, height / 2, int(height / self.resolution) + 1)
        if axis == 'z':
            x, y, z = np.meshgrid(sample, sample, direction_sample)
            mask = x**2 + y**2 <= radius**2
        elif axis == 'x':
            x, y, z = np.meshgrid(direction_sample, sample, sample)
            mask = z**2 + y**2 <= radius**2
        elif axis == 'y':
            x, y, z = np.meshgrid(sample, direction_sample, sample)
            mask = z**2 + x**2 <= radius**2
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")
        self.coords = np.column_stack((x[mask], y[mask], z[mask]))
        self.coords += self.center
        self.update()

    def constraints(self, space):
        if self.axis == 'z':
            distances = np.linalg.norm(space[:, :2] - self.center[:2], axis=1)
            return (distances <= self.radius) & (np.abs(space[:, 2] - self.center[2]) <= (self.height / 2))
        elif self.axis == 'x':
            distances = np.linalg.norm(space[:, [1, 2]] - self.center[[1, 2]], axis=1)
            return (distances <= self.radius) & (np.abs(space[:, 0] - self.center[0]) <= (self.height / 2))
        elif self.axis == 'y':
            distances = np.linalg.norm(space[:, [0, 2]] - self.center[[0, 2]], axis=1)
            return (distances <= self.radius) & (np.abs(space[:, 1] - self.center[1]) <= (self.height / 2))
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

class ArbitraryZone(Zone):
    def __init__(self, resolution=2e-3, coords=None):
        super().__init__(resolution)
        if coords is None:
            coords = np.array([])
        self.coords = np.array(coords, dtype=np.float32)
        self.update()

    def constraints(self, space):
        return self.isin(space, self.coords).any(axis=1)

Zone_LUT = {
    'sphere': SphereZone,
    'cube': CubeZone,
    'rect': RectZone,
    'cylinder': CylinderZone,
}