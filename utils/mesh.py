import gmsh
import os
from shutil import copyfile
import numpy as np
import process.magnet
import utils.zone
from scipy.spatial import Delaunay

class MeshBuilder:
    def __init__(self, roi: utils.zone.Zone, materials=None):
        if materials is None:
            raise ValueError("Materials must be provided for mesh generation.")
        self.dsv = roi
        self.materials = materials
        self.muList = materials[:, 0]
        self.coordList = materials[:, 1:4]
        self.polarizationList = materials[:, 4:7]
        self.sizeList = materials[:, 7]
        self.sizeList = np.power(self.sizeList, 1/3)
        self.coordRoi = roi.coords

    def genMesh(self, name):
        pass

class CubeMeshBuilder(MeshBuilder):
    # Assuming all objects are cubes
    def __init__(self, roi: utils.zone.Zone, materials=None, rotation=None):
        # rotation definition:
        # list with N elements
        # Each element is a list containing rotation operations
        # Each operation is a tuple (axis, angle)
        super().__init__(roi, materials)
        self.rotation = rotation
        if len(self.rotation) != self.materials.shape[0]:
            raise ValueError("Rotation must match the number of materials.")

    def genMesh(self, name):
        #super().genMesh(name)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        #gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
        #gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)

        obj_data = "DefineConstant[\n"

        objs = []
        for i in range(self.materials.shape[0]):
            mu_r = self.muList[i]
            coords = self.coordList[i]
            polarization = self.polarizationList[i]
            size = self.sizeList[i]
            rot = self.rotation[i]
            obj = gmsh.model.occ.addBox(
                coords[0]-size/2, coords[1]-size/2, coords[2]-size/2,
                size, size, size
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
        
        maxsize = np.max(self.sizeList)
        boundarymin = [np.min(self.coordList[:,i])-5*maxsize*np.sqrt(2)/2 for i in range(3)]
        boundarymax = [np.max(self.coordList[:,i])+5*maxsize*np.sqrt(2)/2 for i in range(3)]
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
        gmsh.model.mesh.optimize("Netgen")
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
        #copyfile('../scripts/fem.pro', 'model/fem.pro')
    
    #def genROI(self):
        #print(gmsh.isInitialized())
        #gmsh.initialize()
        ##gmsh.clear()
        #dsv_points = self.dsv.coords
        #tri = Delaunay(dsv_points)
        #points = {}
        #for i, point in enumerate(dsv_points):
            #points[i] = gmsh.model.occ.addPoint(point[0], point[1], point[2])
        #edges = set()
        #faces = set()
        #face_to_tets = {}
        #for tet_idx, simplex in enumerate(tri.simplices):
            #for i in range(4):
                #for j in range(i + 1, 4):
                    #edge = tuple(sorted([simplex[i], simplex[j]]))
                    #edges.add(edge)
            #for i in range(4):
                #face_vertices = [simplex[j] for j in range(4) if j != i]
                #face = tuple(sorted(face_vertices))
                #faces.add(face)
                #if face not in face_to_tets:
                    #face_to_tets[face] = []
                #face_to_tets[face].append(tet_idx)
        #boundary_faces = []
        #internal_faces = []
        #problematic_faces = []
        #for face, tet_list in face_to_tets.items():
            #if len(tet_list) == 1:
                #boundary_faces.append(face)
            #elif len(tet_list) == 2:
                #internal_faces.append(face)
            #else:
                #problematic_faces.append((face, tet_list))
        #degenerate_faces = []
        #for face in faces:
            #p1, p2, p3 = [dsv_points[i] for i in face]
            #v1 = p2 - p1
            #v2 = p3 - p1
            #normal = np.cross(v1, v2)
            #area = np.linalg.norm(normal) / 2
            #if area < 1e-12:
                #degenerate_faces.append((face, area))
        #line_tags = {}
        #for edge in edges:
            #try:
                #line_tags[edge] = gmsh.model.occ.addLine(points[edge[0]], points[edge[1]])
            #except Exception as e:
                #print(f"Failed to create line {edge}: {e}")
        #surface_tags = {}
        #valid_faces = [f for f in faces if f not in [df[0] for df in degenerate_faces]]
        #for face in valid_faces:
            #try:
                #face_edges = []
                #for i in range(3):
                    #edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                    #if edge in line_tags:
                        #face_edges.append(line_tags[edge])
                #if len(face_edges) == 3:
                    #curve_loop = gmsh.model.occ.addCurveLoop(face_edges)
                    #surface_tags[face] = gmsh.model.occ.addPlaneSurface([curve_loop])
            #except Exception as e:
                #print(f"Failed to create surface for face {face}: {e}")
        #dsv = []
        #for tet_idx, simplex in enumerate(tri.simplices):
            #tetra_surfaces = []
            #all_faces_valid = True
            #for i in range(4):
                #face_vertices = [simplex[j] for j in range(4) if j != i]
                #face = tuple(sorted(face_vertices))
                #if face in surface_tags:
                    #tetra_surfaces.append(surface_tags[face])
                #else:
                    #all_faces_valid = False
                    #break
            #if all_faces_valid and len(tetra_surfaces) == 4:
                #try:
                    #surface_loop = gmsh.model.occ.addSurfaceLoop(tetra_surfaces)
                    #dsv.append(gmsh.model.occ.addVolume([surface_loop]))
                #except Exception as e:
                    #print(f"Failed to create volume for tetrahedron {tet_idx}: {e}")
        #gmsh.model.occ.synchronize()
        #gmsh.option.setNumber("Mesh.Algorithm3D", 0)
        #try:
            #gmsh.model.mesh.generate(3)
        #except:
            #self.genROI()
            #return
        #name = "model/roi"
        #gmsh.write(f"{name}.msh")
        #gmsh.write(f"{name}.geo_unrolled")
        #copyfile(f"{name}.geo_unrolled", f"{name}.geo")
        #os.remove(f"{name}.geo_unrolled")
        #gmsh.write(f"{name}.geo.opt")
        #gmsh.finalize()
        #self.roiMesh = name + '.msh'
