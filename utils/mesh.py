import gmsh
import os
from shutil import copyfile
import numpy as np
import process.magnet
import utils.zone
from scipy.spatial import Delaunay

def tetraVolume(ori, p1, p2, p3):
    mat = np.array([
        [p1[0]-ori[0], p2[0]-ori[0], p3[0]-ori[0]],
        [p1[1]-ori[1], p2[1]-ori[1], p3[1]-ori[1]],
        [p1[2]-ori[2], p2[2]-ori[2], p3[2]-ori[2]],
    ])
    return np.abs(np.linalg.det(mat))/6

class MeshBuilder:
    def __init__(self, roi: utils.zone.Zone, materials=None, rotation=None):
        if materials is None:
            raise ValueError("Materials must be provided for mesh generation.")
        self.dsv = roi
        self.materials = materials
        self.muList = materials[:, 0]
        self.coordList = materials[:, 1:4]
        self.polarizationList = materials[:, 4:7]
        self.shape = materials[:, 7]
        self.shapeParam = materials[:, 8:]
        self.coordRoi = roi.coords
        self.rotation = rotation
        if len(self.rotation) != self.materials.shape[0]:
            raise ValueError("Rotation must match the number of materials.")
            
    if False:
        def genMeshFuse(self, name):
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
            gmsh.option.setNumber("Mesh.Optimize", 0)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

            obj_data = "DefineConstant[\n"

            objs = []
            mags = []
            irons = []
            iron_mur = 0
            for i in range(self.materials.shape[0]):
                mu_r = self.muList[i]
                coords = self.coordList[i]
                polarization = self.polarizationList[i]
                shape = self.shape[i]
                shapeParam = self.shapeParam[i]
                rot = self.rotation[i]
                if shape == 0:  # cube
                    size = shapeParam[0]
                    obj = gmsh.model.occ.addBox(
                        coords[0]-size/2, coords[1]-size/2, coords[2]-size/2,
                        size, size, size
                    )
                elif shape == 1:  # cuboid
                    size = shapeParam
                    obj = gmsh.model.occ.addBox(
                        coords[0]-size[0]/2, coords[1]-size[1]/2, coords[2]-size[2]/2,
                        size[0], size[1], size[2]
                    )
                elif shape == 2:  # cylinder
                    radius = shapeParam[0]
                    height = shapeParam[1]
                    faceto = shapeParam[2]  # 0: x, 1: y, 2: z
                    if faceto == 0:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0]-height/2, coords[1], coords[2],
                            height, 0, 0,
                            radius
                        )
                    elif faceto == 1:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1]-height/2, coords[2],
                            0, height, 0,
                            radius
                        )
                    else:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1], coords[2]-height/2,
                            0, 0, height,
                            radius
                        )
                elif shape == 3:  # fan with inner hole
                    inner_radius = shapeParam[0]
                    outer_radius = shapeParam[1]
                    height = shapeParam[2]
                    angle_start = shapeParam[3]
                    angle_end = shapeParam[4]
                    faceto = shapeParam[5]  # 0: x, 1: y, 2: z
                    angle = angle_end - angle_start
                    if faceto == 0:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0]-height/2, coords[1], coords[2],
                            height, 0, 0,
                            outer_radius,
                            angle=angle
                        )
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0]-height/2, coords[1], coords[2],
                            height, 0, 0,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                        gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 1, 0, 0, angle_start)
                    elif faceto == 1:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1]-height/2, coords[2],
                            0, height, 0,
                            outer_radius,
                            angle=angle
                        )
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1]-height/2, coords[2],
                            0, height, 0,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                        gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 0, 1, 0, angle_start)
                    else:
                        obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1], coords[2]-height/2,
                            0, 0, height,
                            outer_radius,
                            angle=angle
                        )
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1], coords[2]-height/2,
                            0, 0, height,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                        gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 0, 0, 1, angle_start)
                elif shape == 4:  # triangle prism
                    inner_x = shapeParam[0]; inner_z = shapeParam[1]
                    offset_x = shapeParam[2]; offset_z = shapeParam[3]
                    length = shapeParam[4]
                    p1x = inner_x; p1z = inner_z
                    p2x = inner_x + offset_x; p2z = inner_z
                    p3x = inner_x; p3z = inner_z + offset_z
                    p1y = -length/2; p2y = -length/2; p3y = -length/2
                    p1 = gmsh.model.occ.addPoint(p1x, p1y, p1z); p2 = gmsh.model.occ.addPoint(p2x, p2y, p2z); p3 = gmsh.model.occ.addPoint(p3x, p3y, p3z)
                    l1 = gmsh.model.occ.addLine(p1, p2); l2 = gmsh.model.occ.addLine(p2, p3); l3 = gmsh.model.occ.addLine(p3, p1)
                    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                    surface = gmsh.model.occ.addPlaneSurface([loop])
                    obj = gmsh.model.occ.extrude([(2, surface)], 0, length, 0)[1][1]
                objs.append(obj)
                if mu_r > 100:
                    irons.append(obj)
                    iron_mur = mu_r
                else:
                    mags.append([obj, 0, mu_r, polarization])
            gmsh.model.occ.synchronize()

            if type(self.dsv) is utils.zone.SphereZone:
                center = self.dsv.center
                radius = self.dsv.radius
                dsv = gmsh.model.occ.addSphere(
                    center[0], center[1], center[2], radius
                )
            else:
                raise Exception("Unsupported zone type for Halbach mesh generation.")
        
            if len(irons) != 0:
                iron = irons[0]
                for i in irons[1:]:
                    iron = gmsh.model.occ.fuse([(3, iron)], [(3, i)], removeObject=True, removeTool=True)[0][0][1]
                gmsh.model.occ.synchronize()

            boundarymin = [np.min(self.coordList[:,i])-2 for i in range(3)]
            boundarymax = [np.max(self.coordList[:,i])+2 for i in range(3)]
            bbox = gmsh.model.occ.addBox(
                boundarymin[0], boundarymin[1], boundarymin[2],
                boundarymax[0]-boundarymin[0], boundarymax[1]-boundarymin[1], boundarymax[2]-boundarymin[2]
            )
            gmsh.model.occ.synchronize()
            bbox_boundary = gmsh.model.getBoundary([[3, bbox]], oriented=False)
            air_domain = gmsh.model.occ.cut([(3, bbox)], [(3, dsv)], removeObject=True, removeTool=False)[0][0][1]
            air_domain = gmsh.model.occ.cut([(3, air_domain)], [(3, mag[0]) for mag in mags], removeObject=True, removeTool=False)[0][0][1]
            if len(irons) != 0:
                air_domain = gmsh.model.occ.cut([(3, air_domain)], [(3, iron)], removeObject=True, removeTool=False)[0][0][1]
            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

            count = 0
            mag_tags = []
            for o in mags:
                count += 1
                o_points = gmsh.model.getBoundary([(3, o[0])], combined=False, oriented=False, recursive=True)
                for point in o_points:
                    gmsh.model.mesh.setSize([point], self.dsv.resolution*2)
                obj_tag = gmsh.model.addPhysicalGroup(3, [o[0]], count)
                gmsh.model.setPhysicalName(3, obj_tag, f"Obj_{count}")
                obj_data += "angle_" + str(obj_tag) + " = " + str(o[1]) + "\n"
                obj_data += "mu_r_" + str(obj_tag) + " = " + str(o[2]) + "\n"
                obj_data += "polarization_x_" + str(obj_tag) + " = " + str(o[3][0]) + "\n"
                obj_data += "polarization_y_" + str(obj_tag) + " = " + str(o[3][1]) + "\n"
                obj_data += "polarization_z_" + str(obj_tag) + " = " + str(o[3][2]) + "\n"
                mag_tags.append(obj_tag)
            if len(irons) != 0:
                count += 1
                o_points = gmsh.model.getBoundary([(3, iron)], combined=False, oriented=False, recursive=True)
                for point in o_points:
                    gmsh.model.mesh.setSize([point], self.dsv.resolution*2)
                iron_tag = gmsh.model.addPhysicalGroup(3, [iron], count)
                obj_data += "angle_" + str(iron_tag) + " = " + str(0) + "\n"
                obj_data += "mu_r_" + str(iron_tag) + " = " + str(iron_mur) + "\n"
                obj_data += "polarization_x_" + str(iron_tag) + " = " + str(0) + "\n"
                obj_data += "polarization_y_" + str(iron_tag) + " = " + str(0) + "\n"
                obj_data += "polarization_z_" + str(iron_tag) + " = " + str(0) + "\n"
                gmsh.model.setPhysicalName(3, iron_tag, f"Obj_{count}")
            gmsh.model.occ.synchronize()

            obj_data += "matNum = " + str(count) + "\n"
            obj_data += "];\n"
            obj_data += "Include \"fem.pro\"\n"
            with open(f"{name}.pro", "w") as f:
                f.write(obj_data)

            dsv_tag = gmsh.model.addPhysicalGroup(3, [dsv])
            gmsh.model.setPhysicalName(3, dsv_tag, "DSV")
            air_tag = gmsh.model.addPhysicalGroup(3, [air_domain])
            gmsh.model.setPhysicalName(3, air_tag, "Air")
            boundary_tag = gmsh.model.addPhysicalGroup(2, [x[1] for x in bbox_boundary])
            gmsh.model.setPhysicalName(2, boundary_tag, "Boundary")

            # enforce fine mesh in DSV!!!
            dsv_points = gmsh.model.getBoundary([(3, dsv)], combined=False, oriented=False, recursive=True)
            for point in dsv_points:
                gmsh.model.mesh.setSize([point], self.dsv.resolution)

            gmsh.model.mesh.generate(3)
            gmsh.write(f"{name}.msh")
            gmsh.finalize()
            # xxx
            with open('../scripts/fem.pro') as f:
                profile = f.read()
                profile = self.dsv.genMeshPoint(profile)
                with open('model/fem.pro', 'w') as f:
                    f.write(profile)

    def genMesh(self, name, full_output=False, res_factor=1):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        #gmsh.option.setNumber("Mesh.MeshSizeMax", self.dsv.resolution*10*res_factor)

        obj_data = "DefineConstant[\n"

        objs = []
        for i in range(self.materials.shape[0]):
            mu_r = self.muList[i]
            coords = self.coordList[i]
            polarization = self.polarizationList[i]
            shape = self.shape[i]
            shapeParam = self.shapeParam[i]
            rot = self.rotation[i]
            if shape == 0:  # cube
                size = shapeParam[0]
                obj = gmsh.model.occ.addBox(
                    coords[0]-size/2, coords[1]-size/2, coords[2]-size/2,
                    size, size, size
                )
            elif shape == 1:  # cuboid
                size = shapeParam
                obj = gmsh.model.occ.addBox(
                    coords[0]-size[0]/2, coords[1]-size[1]/2, coords[2]-size[2]/2,
                    size[0], size[1], size[2]
                )
            elif shape == 2:  # cylinder
                radius = shapeParam[0]
                height = shapeParam[1]
                faceto = shapeParam[2]  # 0: x, 1: y, 2: z
                if faceto == 0:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0]-height/2, coords[1], coords[2],
                        height, 0, 0,
                        radius
                    )
                elif faceto == 1:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0], coords[1]-height/2, coords[2],
                        0, height, 0,
                        radius
                    )
                else:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0], coords[1], coords[2]-height/2,
                        0, 0, height,
                        radius
                    )
            elif shape == 3:  # fan with inner hole
                inner_radius = shapeParam[0]
                outer_radius = shapeParam[1]
                height = shapeParam[2]
                angle_start = shapeParam[3]
                angle_end = shapeParam[4]
                faceto = shapeParam[5]  # 0: x, 1: y, 2: z
                angle = angle_end - angle_start
                if faceto == 0:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0]-height/2, coords[1], coords[2],
                        height, 0, 0,
                        outer_radius,
                        angle=angle
                    )
                    if inner_radius > 0:
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0]-height/2, coords[1], coords[2],
                            height, 0, 0,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                    gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 1, 0, 0, angle_start)
                elif faceto == 1:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0], coords[1]-height/2, coords[2],
                        0, height, 0,
                        outer_radius,
                        angle=angle
                    )
                    if inner_radius > 0:
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1]-height/2, coords[2],
                            0, height, 0,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                    gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 0, 1, 0, angle_start)
                else:
                    obj = gmsh.model.occ.addCylinder(
                        coords[0], coords[1], coords[2]-height/2,
                        0, 0, height,
                        outer_radius,
                        angle=angle
                    )
                    if inner_radius > 0:
                        inner_obj = gmsh.model.occ.addCylinder(
                            coords[0], coords[1], coords[2]-height/2,
                            0, 0, height,
                            inner_radius,
                            angle=angle
                        )
                        obj = gmsh.model.occ.cut([(3, obj)], [(3, inner_obj)], removeObject=True, removeTool=True)[0][0][1]
                    gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], 0, 0, 1, angle_start)
            elif shape == 4:  # triangle prism to y
                inner_x = shapeParam[0]; inner_z = shapeParam[1]
                offset_x = shapeParam[2]; offset_z = shapeParam[3]
                length = shapeParam[4]; center_y = shapeParam[5]
                p1x = inner_x; p1z = inner_z
                p2x = inner_x + offset_x; p2z = inner_z
                p3x = inner_x; p3z = inner_z + offset_z
                p1y = center_y - length/2; p2y = center_y - length/2; p3y = center_y - length/2
                p1 = gmsh.model.occ.addPoint(p1x, p1y, p1z); p2 = gmsh.model.occ.addPoint(p2x, p2y, p2z); p3 = gmsh.model.occ.addPoint(p3x, p3y, p3z)
                l1 = gmsh.model.occ.addLine(p1, p2); l2 = gmsh.model.occ.addLine(p2, p3); l3 = gmsh.model.occ.addLine(p3, p1)
                loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                surface = gmsh.model.occ.addPlaneSurface([loop])
                obj = gmsh.model.occ.extrude([(2, surface)], 0, length, 0)[1][1]
            elif shape == 5:  # triangle prism to z
                inner_x = shapeParam[0]; inner_y = shapeParam[1]
                offset_x = shapeParam[2]; offset_y = shapeParam[3]
                length = shapeParam[4]; center_z = shapeParam[5]
                p1x = inner_x; p1y = inner_y
                p2x = inner_x + offset_x; p2y = inner_y
                p3x = inner_x; p3y = inner_y + offset_y
                p1z = center_z - length/2; p2z = center_z - length/2; p3z = center_z - length/2
                p1 = gmsh.model.occ.addPoint(p1x, p1y, p1z); p2 = gmsh.model.occ.addPoint(p2x, p2y, p2z); p3 = gmsh.model.occ.addPoint(p3x, p3y, p3z)
                l1 = gmsh.model.occ.addLine(p1, p2); l2 = gmsh.model.occ.addLine(p2, p3); l3 = gmsh.model.occ.addLine(p3, p1)
                loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                surface = gmsh.model.occ.addPlaneSurface([loop])
                obj = gmsh.model.occ.extrude([(2, surface)], 0, 0, length)[1][1]
            elif shape == 6:  # triangle prism to x
                inner_y = shapeParam[0]; inner_z = shapeParam[1]
                offset_y = shapeParam[2]; offset_z = shapeParam[3]
                length = shapeParam[4]; center_x = shapeParam[5]
                p1y = inner_y; p1z = inner_z
                p2y = inner_y + offset_y; p2z = inner_z
                p3y = inner_y; p3z = inner_z + offset_z
                p1x = center_x - length/2; p2x = center_x - length/2; p3x = center_x - length/2
                p1 = gmsh.model.occ.addPoint(p1x, p1y, p1z); p2 = gmsh.model.occ.addPoint(p2x, p2y, p2z); p3 = gmsh.model.occ.addPoint(p3x, p3y, p3z)
                l1 = gmsh.model.occ.addLine(p1, p2); l2 = gmsh.model.occ.addLine(p2, p3); l3 = gmsh.model.occ.addLine(p3, p1)
                loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                surface = gmsh.model.occ.addPlaneSurface([loop])
                obj = gmsh.model.occ.extrude([(2, surface)], length, 0, 0)[1][1]
            elif shape == 7:  # tetra along z
                origin_x = coords[0]; origin_y = coords[1]; origin_z = coords[2]
                axis = shapeParam[5]  # 0: x, 1: y, 2: z
                if axis == 0:
                    p1x = shapeParam[0]; p1y = origin_y; p1z = origin_z
                    p2y = shapeParam[1]; p2z = shapeParam[2]; p2x = p1x
                    p3y = shapeParam[3]; p3z = shapeParam[4]; p3x = p1x
                elif axis == 1:
                    p1y = shapeParam[0]; p1x = origin_x; p1z = origin_z
                    p2x = shapeParam[1]; p2z = shapeParam[2]; p2y = p1y
                    p3x = shapeParam[3]; p3z = shapeParam[4]; p3y = p1y
                elif axis == 2:
                    p1z = shapeParam[0]; p1x = origin_x; p1y = origin_y
                    p2x = shapeParam[1]; p2y = shapeParam[2]; p2z = p1z
                    p3x = shapeParam[3]; p3y = shapeParam[4]; p3z = p1z
                p1 = gmsh.model.occ.addPoint(origin_x, origin_y, origin_z)
                p2 = gmsh.model.occ.addPoint(p1x, p1y, p1z)
                p3 = gmsh.model.occ.addPoint(p2x, p2y, p2z)
                p4 = gmsh.model.occ.addPoint(p3x, p3y, p3z)
                l1 = gmsh.model.occ.addLine(p1, p2)
                l2 = gmsh.model.occ.addLine(p2, p3)
                l3 = gmsh.model.occ.addLine(p3, p1)
                l4 = gmsh.model.occ.addLine(p1, p4)
                l5 = gmsh.model.occ.addLine(p2, p4)
                l6 = gmsh.model.occ.addLine(p3, p4)
                loop1 = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                s1 = gmsh.model.occ.addPlaneSurface([loop1])
                loop2 = gmsh.model.occ.addCurveLoop([l1, l5, -l4])
                s2 = gmsh.model.occ.addPlaneSurface([loop2])
                loop3 = gmsh.model.occ.addCurveLoop([l3, l4, -l6])
                s3 = gmsh.model.occ.addPlaneSurface([loop3])
                loop4 = gmsh.model.occ.addCurveLoop([l2, l6, -l5])
                s4 = gmsh.model.occ.addPlaneSurface([loop4])
                surf = gmsh.model.occ.addSurfaceLoop([s1, s2, s3, s4])
                obj = gmsh.model.occ.addVolume([surf])
            objs.append(obj)
            if len(rot) > 0:
                for axis, angle in rot:
                    gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], axis[0], axis[1], axis[2], angle)
            obj_data += "angle_" + str(obj) + " = " + str(0) + "\n"
            obj_data += "mu_r_" + str(obj) + " = " + str(mu_r) + "\n"
            obj_data += "polarization_x_" + str(obj) + " = " + str(polarization[0]) + "\n"
            obj_data += "polarization_y_" + str(obj) + " = " + str(polarization[1]) + "\n"
            obj_data += "polarization_z_" + str(obj) + " = " + str(polarization[2]) + "\n"
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        if type(self.dsv) is utils.zone.SphereZone:
            center = self.dsv.center
            radius = self.dsv.radius
            dsv = gmsh.model.occ.addSphere(
                center[0], center[1], center[2], radius
            )
        else:
            raise Exception("Unsupported zone type for Halbach mesh generation.")
        
        boundarymin = [np.min(self.coordList[:,i])-2 for i in range(3)]
        boundarymax = [np.max(self.coordList[:,i])+2 for i in range(3)]
        #boundarymin = [np.min(self.coordList[:,i])-0.1 for i in range(3)]
        #boundarymax = [np.max(self.coordList[:,i])+0.1 for i in range(3)]
        bbox = gmsh.model.occ.addBox(
            boundarymin[0], boundarymin[1], boundarymin[2],
            boundarymax[0]-boundarymin[0], boundarymax[1]-boundarymin[1], boundarymax[2]-boundarymin[2]
        )
        #bbox = gmsh.model.occ.addBox(
            #-6, -6, -6,
            #12, 12, 12
        #)
        gmsh.model.occ.synchronize()
        bbox_boundary = gmsh.model.getBoundary([[3, bbox]], oriented=False)

        bbox_cut_dsv = gmsh.model.occ.cut([(3, bbox)], [(3, dsv)], removeObject=True, removeTool=False)[0][0][1]
        air_domain = gmsh.model.occ.cut([(3, bbox_cut_dsv)], [(3, o) for o in objs], removeObject=True, removeTool=False)[0][0][1]
        gmsh.model.occ.synchronize()
        
        count = 0
        obj_tags = []
        for o in objs:
            count += 1
            o_points = gmsh.model.getBoundary([(3, o)], combined=False, oriented=False, recursive=True)
            for point in o_points:
                gmsh.model.mesh.setSize([point], self.dsv.resolution*2*res_factor)
            obj_tag = gmsh.model.addPhysicalGroup(3, [o], count)
            gmsh.model.setPhysicalName(3, obj_tag, f"Obj_{count}")
            obj_tags.append(obj_tag)
        gmsh.model.occ.synchronize()
        obj_data += "matNum = " + str(count) + "\n"
        obj_data += "];\n"
        obj_data += "Include \"fem.pro\"\n"
        with open(f"{name}.pro", "w") as f:
            f.write(obj_data)
        dsv_tag = gmsh.model.addPhysicalGroup(3, [dsv])
        gmsh.model.setPhysicalName(3, dsv_tag, "DSV")
        air_tag = gmsh.model.addPhysicalGroup(3, [air_domain])
        gmsh.model.setPhysicalName(3, air_tag, "Air")
        boundary_tag = gmsh.model.addPhysicalGroup(2, [x[1] for x in bbox_boundary])
        gmsh.model.setPhysicalName(2, boundary_tag, "Boundary")

        # enforce fine mesh in DSV!!!
        dsv_points = gmsh.model.getBoundary([(3, dsv)], combined=False, oriented=False, recursive=True)
        for point in dsv_points:
            gmsh.model.mesh.setSize([point], self.dsv.resolution)

        gmsh.model.mesh.generate(3)
        #gmsh.model.mesh.optimize("Netgen")
        gmsh.write(f"{name}.msh")
        if full_output:
            gmsh.write(f"{name}.geo_unrolled")
            copyfile(f"{name}.geo_unrolled", f"{name}.geo")
            os.remove(f"{name}.geo_unrolled")
            gmsh.write(f"{name}.geo.opt")
            gmsh.write(f"{name}.step")
        gmsh.finalize()
        if full_output:
            with open('../scripts/fem_result.pro') as f:
                profile = f.read()
                profile = self.dsv.genMeshPoint(profile)
                with open('model/fem.pro', 'w') as f:
                    f.write(profile)
        else:
            with open('../scripts/fem.pro') as f:
                profile = f.read()
                profile = self.dsv.genMeshPoint(profile)
                with open('model/fem.pro', 'w') as f:
                    f.write(profile)
            

class CubeMeshBuilder(MeshBuilder):
    # Assuming all objects are cubes
    def __init__(self, roi: utils.zone.Zone, materials=None, rotation=None):
        # rotation definition:
        # list with N elements
        # Each element is a list containing rotation operations
        # Each operation is a tuple (axis, angle)
        if materials is None:
            raise ValueError("Materials must be provided for mesh generation.")
        self.dsv = roi
        self.isCube = materials.shape[1] == 8
        self.materials = materials
        self.muList = materials[:, 0]
        self.coordList = materials[:, 1:4]
        self.polarizationList = materials[:, 4:7]
        if self.isCube:
            self.sizeList = materials[:, 7]
            self.sizeList = np.power(self.sizeList, 1/3)
        else:
            self.sizeList = materials[:, 7:10]
        self.coordRoi = roi.coords
        self.rotation = rotation
        if len(self.rotation) != self.materials.shape[0]:
            raise ValueError("Rotation must match the number of materials.")

    def genMesh(self, name):
        #super().genMesh(name)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        #gmsh.option.setNumber("Mesh.RandomSeed", 0)
        #gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
        #gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)

        obj_data = "DefineConstant[\n"

        objs = []
        for i in range(self.materials.shape[0]):
            mu_r = self.muList[i]
            coords = self.coordList[i]
            polarization = self.polarizationList[i]
            size = self.sizeList[i]
            rot = self.rotation[i]
            if self.isCube:
                obj = gmsh.model.occ.addBox(
                    coords[0]-size/2, coords[1]-size/2, coords[2]-size/2,
                    size, size, size
                )
            else:
                obj = gmsh.model.occ.addBox(
                    coords[0]-size[0]/2, coords[1]-size[1]/2, coords[2]-size[2]/2,
                    size[0], size[1], size[2]
                )
            objs.append(obj)
            if len(rot) > 0:
                for axis, angle in rot:
                    gmsh.model.occ.rotate([(3, obj)], coords[0], coords[1], coords[2], axis[0], axis[1], axis[2], angle)
            obj_data += "angle_" + str(obj) + " = " + str(0) + "\n"
            obj_data += "mu_r_" + str(obj) + " = " + str(mu_r) + "\n"
            #obj_data += "polarization_" + str(obj) + " = Vector[" + ", ".join(map(str, polarization)) + "]\n"
            obj_data += "polarization_" + str(obj) + " = " + str(polarization[2]) + "\n"
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        if type(self.dsv) is utils.zone.SphereZone:
            center = self.dsv.center
            radius = self.dsv.radius
            dsv = gmsh.model.occ.addSphere(
                center[0], center[1], center[2], radius
            )
        else:
            raise Exception("Unsupported zone type for Halbach mesh generation.")
        
        maxsize = np.max(self.sizeList, axis=0)
        if self.isCube:
            boundarymin = [np.min(self.coordList[:,i])-5*maxsize[i]*np.sqrt(2)/2 for i in range(3)]
            boundarymax = [np.max(self.coordList[:,i])+5*maxsize[i]*np.sqrt(2)/2 for i in range(3)]
        else:
            boundarymin = [np.min(self.coordList[:,i])-maxsize[i]/4 for i in range(3)]
            boundarymax = [np.max(self.coordList[:,i])+maxsize[i]/4 for i in range(3)]
            #boundarymin = [np.min(self.coordList[:,i])-0.1 for i in range(3)]
            #boundarymax = [np.max(self.coordList[:,i])+0.1 for i in range(3)]
        bbox = gmsh.model.occ.addBox(
            boundarymin[0], boundarymin[1], boundarymin[2],
            boundarymax[0]-boundarymin[0], boundarymax[1]-boundarymin[1], boundarymax[2]-boundarymin[2]
        )
        #bbox = gmsh.model.occ.addBox(
            #-6, -6, -6,
            #12, 12, 12
        #)
        gmsh.model.occ.synchronize()
        bbox_boundary = gmsh.model.getBoundary([[3, bbox]], oriented=False)

        bbox_cut_dsv = gmsh.model.occ.cut([(3, bbox)], [(3, dsv)], removeObject=True, removeTool=False)[0][0][1]
        air_domain = gmsh.model.occ.cut([(3, bbox_cut_dsv)], [(3, o) for o in objs], removeObject=True, removeTool=False)[0][0][1]
        gmsh.model.occ.synchronize()

        count = 0
        obj_tags = []
        for o in objs:
            count += 1
            o_points = gmsh.model.getBoundary([(3, o)], combined=False, oriented=False, recursive=True)
            for point in o_points:
                gmsh.model.mesh.setSize([point], self.dsv.resolution*2)
            obj_tag = gmsh.model.addPhysicalGroup(3, [o], count)
            gmsh.model.setPhysicalName(3, obj_tag, f"Obj_{count}")
            obj_tags.append(obj_tag)
        gmsh.model.occ.synchronize()
        obj_data += "matNum = " + str(count) + "\n"
        obj_data += "];\n"
        obj_data += "Include \"fem.pro\"\n"
        with open(f"{name}.pro", "w") as f:
            f.write(obj_data)
        dsv_tag = gmsh.model.addPhysicalGroup(3, [dsv])
        gmsh.model.setPhysicalName(3, dsv_tag, "DSV")
        air_tag = gmsh.model.addPhysicalGroup(3, [air_domain])
        gmsh.model.setPhysicalName(3, air_tag, "Air")
        boundary_tag = gmsh.model.addPhysicalGroup(2, [x[1] for x in bbox_boundary])
        gmsh.model.setPhysicalName(2, boundary_tag, "Boundary")

        # enforce fine mesh in DSV!!!
        dsv_points = gmsh.model.getBoundary([(3, dsv)], combined=False, oriented=False, recursive=True)
        for point in dsv_points:
            gmsh.model.mesh.setSize([point], self.dsv.resolution)

        gmsh.model.mesh.generate(3)
        #gmsh.model.mesh.optimize("Netgen")
        gmsh.write(f"{name}.msh")
        gmsh.write(f"{name}.geo_unrolled")
        copyfile(f"{name}.geo_unrolled", f"{name}.geo")
        os.remove(f"{name}.geo_unrolled")
        gmsh.write(f"{name}.geo.opt")
        gmsh.finalize()
        # xxx
        with open('../scripts/fem.pro') as f:
            profile = f.read()
            profile = self.dsv.genMeshPoint(profile)
            with open('model/fem.pro', 'w') as f:
                f.write(profile)