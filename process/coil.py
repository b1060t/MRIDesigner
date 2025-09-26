import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
import utils.zone

class CoilCollection:
    def __init__(self):
        self.coillist = []
        self.collection = magpy.Collection()

    def addCoil(self, coil):
        self.coillist.append(coil)
        self.collection.add(coil)

    def removeCoil(self, coil):
        if coil in self.coillist:
            self.coillist.remove(coil)
            self.collection.remove(coil)
        else:
            raise ValueError("Coil not found in collection")

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
        elements = [e for e in elements if isinstance(e, magpy.current.Circle)]
        pos = np.array([e.position for e in elements])
        rot = np.array([e.orientation.as_euler('xyz') for e in elements])
        return pos, rot

    def getB(self, roi: utils.zone.Zone):
        sensor = magpy.Sensor(pixel=roi.coords)
        return self.collection.getB(sensor)

    def getBs(self, roi: utils.zone.Zone):
        sensor = magpy.Sensor(pixel=roi.coords)
        fields = []
        for coil in self.coillist:
            coil_field = coil.getB(sensor)
            fields.append(coil_field)
        fields = np.array(fields)
        return fields

    def getPos(self):
        positions = list(map(lambda c: c.position, self.collection.children_all))
        return np.array(positions)