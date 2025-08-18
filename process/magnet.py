import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
import utils.zone
import gmsh

def checkOverlap(rl, size, num=0):
    rs = size / np.sqrt(2)
    n = int(np.floor(np.pi / np.arcsin(rs / rl))) - 3
    return num > n, n

def getB_aaa(roi: utils.zone.Zone, ironList=None, magList=None):
    muiron = ironList[:, 0][:, np.newaxis]
    muiron = np.repeat(muiron, 3, axis=0)
    coordiron = ironList[:, 1:4]
    polarizationiron = ironList[:, 4:7]
    sizeiron = ironList[:, 7][:, np.newaxis]
    magiron = magList[:, 0][:, np.newaxis]
    magiron = np.repeat(magiron, 3, axis=0)
    coordmag = magList[:, 1:4]
    polarizationmag = magList[:, 4:7]
    sizemag = magList[:, 7][:, np.newaxis]
    G_shape = coordmag.shape[0]
    F_shape = coordiron.shape[0]
    mu_0 = 4 * np.pi * 1e-7
    r_ij = coordiron[:, np.newaxis, :] - coordiron[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    F = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (120*np.pi)
    F[self_mask] = self_value[np.newaxis, :]
    F = F.transpose(0, 2, 1, 3).reshape(3*F_shape, 3*F_shape)
    r_ij = coordiron[:, np.newaxis, :] - coordmag[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    G = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (120*np.pi)
    G[self_mask] = self_value[np.newaxis, :]
    G = G.transpose(0, 2, 1, 3).reshape(3*F_shape, 3*G_shape)
    M0 = (polarizationmag * sizemag).reshape(-1, 1) / mu_0
    I = np.eye(3*F_shape,3*F_shape)
    A = (muiron / (muiron - 1)) * I - F
    b = G @ M0
    M = np.linalg.solve(A, b)

    coordRoi = roi.coords
    r_ij = coordRoi[:, np.newaxis, :] - coordmag[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    D = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (120*np.pi)
    D[self_mask] = self_value[np.newaxis, :]
    D = D.transpose(0, 2, 1, 3).reshape(3*coordRoi.shape[0], 3*coordmag.shape[0])
    field = mu_0 * (D @ M0).reshape(-1, 3)

    r_ij = coordRoi[:, np.newaxis, :] - coordiron[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    D = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (120*np.pi)
    D[self_mask] = self_value[np.newaxis, :]
    D = D.transpose(0, 2, 1, 3).reshape(3*coordRoi.shape[0], 3*coordiron.shape[0])
    field += mu_0 * (D @ M).reshape(-1, 3)

    return field

def getB_mom(roi: utils.zone.Zone, materialList=None):
    # materialList columns:
    # 0: mu_r
    # 1-3: coords
    # 4-6: polarization
    # 7: size

    if materialList is None:
        return ValueError("Material list must be provided for getB_mom.")
    sizeList = materialList[:, 7][:, np.newaxis]
    factor = 1e2 / np.power(np.min(sizeList), 1/3)
    sizeList *= factor ** 3
    muList = materialList[:, 0][:, np.newaxis]
    muList = np.repeat(muList, 3, axis=0)
    coordList = materialList[:, 1:4] * factor
    polarizationList = materialList[:, 4:7]
    coordRoi = roi.coords * factor

    J = coordRoi.shape[0]
    K = coordList.shape[0]
    mu_0 = 4 * np.pi * 1e-7

    r_ij = coordList[:, np.newaxis, :] - coordList[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    F = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (12*np.pi)
    F[self_mask] = self_value[np.newaxis, :]
    F = F.transpose(0, 2, 1, 3).reshape(3*K, 3*K)

    M0 = (polarizationList * sizeList).reshape(-1, 1) / mu_0
    I = np.eye(3*K)
    A = ((muList / (muList - 1)) * I - F) + np.eye(3*K)*1e-12
    b = F @ M0
    M = np.linalg.solve(A, b)

    r_ij = coordRoi[:, np.newaxis, :] - coordList[np.newaxis, :, :]
    r_norm = np.linalg.norm(r_ij, axis=-1)
    r_norm_safe = np.where(r_norm > 1e-10, r_norm, 1.0)
    r_outer = r_ij[:, :, :, np.newaxis] * r_ij[:, :, np.newaxis, :]
    I = np.eye(3)[np.newaxis, np.newaxis, :, :]
    r5 = r_norm_safe[:, :, np.newaxis, np.newaxis]**5
    r3 = r_norm_safe[:, :, np.newaxis, np.newaxis]**3
    D = (3 * r_outer / r5 - I / r3) / (4 * np.pi)
    self_mask = r_norm <= 1e-20
    self_value = np.eye(3) / (12*np.pi)
    D[self_mask] = self_value[np.newaxis, :]
    D = D.transpose(0, 2, 1, 3).reshape(3*J, 3*K)

    field = mu_0 * (D @ (M+M0)).reshape(-1, 3)

    return field

def getB_fem(roi: utils.zone.Zone, materialList=None):
    # materialList columns:
    # 0: mu_r
    # 1-3: coords
    # 4-6: polarization
    # 7: size
    if materialList is None:
        return ValueError("Material list must be provided for getB_fem.")
    
    gmsh.initialize()
    gmsh.model.add("MagneticField")

    for mat in materialList:
        mu_r = mat[0]
        coords = mat[1:4]
        polarization = mat[4:7]
        size = mat[7]
        
        gmsh.model.occ.addPoint(coords[0], coords[1], coords[2])
        gmsh.model.occ.addSphere(coords[0], coords[1], coords[2], size)
    
    roi_coords = roi.coords
    for coord in roi_coords:
        gmsh.model.occ.addPoint(coord[0], coord[1], coord[2])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.finalize()

class MagCollection:
    def __init__(self):
        self.maglist = []
        self.collection = magpy.Collection()

    def addMag(self, mag):
        self.maglist.append(mag)
        self.collection.add(mag)

    def removeMag(self, mag):
        if mag in self.maglist:
            self.maglist.remove(mag)
            self.collection.remove(mag)
        else:
            raise ValueError("Magnet not found in collection.")

    def move(self, offset):
        self.collection.move(offset)

    def setPosition(self, position):
        offset = np.array(position) - self.collection.position
        self.move(offset)
        return self

    def rotate(self, angle, axis='z'):
        rot = R.from_euler(axis, angle, degrees=False)
        self.collection.rotate(rot)

    def getElements(self):
        elements = self.collection.children_all
        elements = [e for e in elements if isinstance(e, magpy.magnet.Cuboid)]
        pos = np.array([e.position for e in elements])
        #rot = np.array([e.orientation for e in elements])
        rot = np.array([e.orientation.as_euler('xyz') for e in elements])
        return pos, rot

    def getB_fast(self, roi: utils.zone.Zone):
        coords = roi.coords
        mu = 4 * np.pi * 1e-7
        field = np.zeros((coords.shape[0], 3), dtype=np.float32)
        
        elements = self.collection.children_all
        elements = [e for e in elements if isinstance(e, magpy.magnet.Cuboid)]
        
        correction_factor = 1e6
        
        for elem in elements:
            pos = elem.position
            
            rotated_polarization = elem.orientation.apply(elem.polarization)
            volume = np.prod(elem.dimension)
            dipole_moment = rotated_polarization * volume * correction_factor
            
            r_vec = coords - pos  # shape: (N, 3)
            
            r_mag = np.linalg.norm(r_vec, axis=1, keepdims=True)
            r_mag_3 = r_mag**3
            r_mag_5 = r_mag**5
            
            mask = (r_mag.flatten() > 1e-10)
            
            # B = (mu/(4*pi)) * (3*(r·m)r/|r|^5 - m/|r|^3)
            r_dot_dipole = np.sum(r_vec * dipole_moment, axis=1, keepdims=True)
            
            field[mask, :] += (mu / (4 * np.pi)) * (
                3 * r_dot_dipole[mask] * r_vec[mask, :] / r_mag_5[mask] - 
                dipole_moment / r_mag_3[mask]
            )

        return field

    def getB(self, roi: utils.zone.Zone):
        sensor = magpy.Sensor(pixel=roi.coords, style_size=2.5, style_opacity = 0.5)
        return self.collection.getB(sensor)

    def getPos(self):
        positions = list(map(lambda c: c.position, self.collection.children_all))
        return np.array(positions)

class HalbachRing(MagCollection):
    def __init__(self, radius=0.1, size=0.01, magNum=10, polarization=1, k=1):
        super().__init__()
        # No overlap assumed here, check outside
        for i in range(magNum):
            angle = 2 * np.pi * i / magNum
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            rot = R.from_euler('z', (k+1)*angle, degrees=False)
            mag = magpy.magnet.Cuboid(
                dimension=(size, size, size),
                orientation=rot,
                position=(x, y, 0),
                polarization=(polarization, 0, 0)
            )
            self.addMag(mag)
    
class Halbach(MagCollection):
    def __init__(self, ringNum=10, radius=0.1, size=0.01, magNum=10, polarization=1, k=1, zpos=None, ringCache=None):
        super().__init__()
        self.meta = {
            'ringNum': ringNum,
            'radius': radius,
            'size': size,
            'magNum': magNum,
            'polarization': polarization,
            'k': k
        }

        for i in range(ringNum):
            pos = (i-ringNum/2) * 0.05 + 0.025
            ring = HalbachRing(radius, size, magNum, polarization, k)
            ring.setPosition((0, 0, pos))
            self.addMag(ring.collection)
        

class RingHalbach(MagCollection):
    def __init__(self, ringNum=10, size=0.01, polarization=1, k=1):
        super().__init__()
        self.meta = {
            'ringNum': ringNum,
            'size': size,
            'polarization': polarization,
            'k': k
        }
        self.ringNum = ringNum

    def addRing(self, radius, size, magNum, polarization, k, zpos):
        ring = HalbachRing(radius, size, magNum, polarization, k)
        ring.setPosition((0, 0, zpos))
        self.addMag(ring.collection)

class PermanentMagnet(MagCollection):
    def __init__(self, radius=0.1, height=0.1, polarization=1, gap=0.5):
        super().__init__()
        self.meta = {
            'radius': radius,
            'height': height,
            'gap': gap,
            'polarization': polarization
        }

        upper = magpy.magnet.Cylinder(
            dimension=(radius, height),
            position=(0, 0, gap/2+height/2),
            polarization=(polarization, 0, 0)
        )
        self.addMag(upper)
        lower = magpy.magnet.Cylinder(
            dimension=(radius, height),
            position=(0, 0, -gap/2-height/2),
            polarization=(polarization, 0, 0)
        )
        self.addMag(lower)

    def getB_fast(self, roi: utils.zone.Zone):
        return Exception("Fast B-field calculation not implemented for PermanentMagnet.")

    def getB_fem(self, roi: utils.zone.Zone):
        gmsh.initialize()
        #gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("PermanentMagnet")

        for mag in self.maglist:
            if isinstance(mag, magpy.magnet.Cylinder):
                radius, height = mag.dimension
                pos = mag.position
                polarization = mag.polarization
                gmsh.model.occ.addCylinder(pos[0], pos[1], pos[2], 0, 0, height, radius)
        
        roi_coords = roi.coords
        for coord in roi_coords:
            gmsh.model.occ.addPoint(coord[0], coord[1], coord[2])

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

        gmsh.finalize()