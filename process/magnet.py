import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
import utils.zone

def checkOverlap(rl, size, num):
    rs = size / np.sqrt(2)
    n = np.floor(np.pi / np.arcsin(rs / rl))
    return num <= n

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