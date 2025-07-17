import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
import utils.zone

def checkOverlap(rl, rs, num):
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

    def rotate(self, angle, axis='z'):
        rot = R.from_euler(axis, angle, degrees=False)
        self.collection.rotate(rot)

    def getB(self, roi: utils.zone.Zone):
        sensor = magpy.Sensor(pixel=roi.coords, style_size=2.5, style_opacity = 0.5)
        return self.collection.getB(sensor)

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
    def __init__(self, ringNum=10, radius=0.1, size=0.01, magNum=10, polarization=1, k=1, zpos=None):
        super().__init__()
        for i in range(ringNum):
            pos = (i-ringNum/2) * 0.05 + 0.025
            ring = HalbachRing(radius, size, magNum, polarization, k)
            ring.setPosition((0, 0, pos))
            self.addMag(ring.collection)