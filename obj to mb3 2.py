import numpy as np
import math
import os
import time
import random
from sys import maxsize
from PIL import Image
from copy import deepcopy
from time import perf_counter


def export(array, pos):
    path = os.path.dirname(os.path.abspath(__file__))
    if path + "\\build" not in [x[0] for x in os.walk(path)]:
        os.makedirs(path + "\\build")
    path += "\\build"
    exact_pos = [math.floor((pos[0] + (model_scale / 2 - center)) / 16),
                 math.floor((pos[1] + (model_scale / 2 - center)) / 16),
                 math.floor((pos[2] + (model_scale / 2 - center)) / 16)]
    print(exact_pos, pos)
    f = open("{}\\{}-{}-{}.mb3d".format(path, exact_pos[0], exact_pos[1], exact_pos[2]), "w")
    line = ""
    for x in range(0, len(array)):
        for y in range(0, len(array[x])):
            for z in range(0, len(array[x, y])):
                line += array[z, x, y].decode('UTF-8') + ","
    line = line[:-1]
    f.write(line)
    f.close()


def texture(points):
    #tex = np.zeros(3)
    #alp = np.zeros(3)
    #    avr_tex = np.zeros(3)
    #    avr_alp = np.zeros(3)
    #    for p in range(0, len(points[f])-1):
    #        avr_tex = avr_tex + data32[int(points[f][p][0])][int(points[f][p][1])]
    #        avr_alp = avr_alp + alpha[int(points[f][p][0])][int(points[f][p][1])]
    #    avr_tex = np.floor(avr_tex / len(points[f]))
    #    avr_alp = np.floor(avr_alp / len(points[f]))
    #    tex = tex + avr_tex
    #    alp = alp + avr_alp
    # tex = np.floor(tex / len(points))
    # alp = np.floor(alp / len(points)) / 255
    randlist = []
    for _ in range(0, data32[int(points[0][1])][int(points[0][0])][0]):
        randlist.append(np.asarray([255, 0, 0]))
    for _ in range(0, data32[int(points[0][1])][int(points[0][0])][1]):
        randlist.append(np.asarray([0, 255, 0]))
    for _ in range(0, data32[int(points[0][1])][int(points[0][0])][2]):
        randlist.append(np.asarray([0, 0, 255]))
    for _ in range(len(randlist), 255):
        randlist.append(np.asarray([0, 0, 0]))
    tex = randlist[random.randint(0, len(randlist) - 1)]
    # tex = data32[int(points[f][0][1])][int(points[f][0][0])]
    alp = alpha[int(points[0][1])][int(points[0][0])] / 255
    # data33[int(points[f][0][0])][int(points[f][0][1])][0] = 0
    # data33[int(points[f][0][0])][int(points[f][0][1])][1] = 255
    # data33[int(points[f][0][0])][int(points[f][0][1])][2] = 0
    if alp[0] <= 0.5:
        ret = "s"
    else:
        ret = "t"

    return ret + str(np.sum(tex * np.array([256 ** 2, 256, 1])))


def arrayinlist(lst, rry):
    for p in lst:
        if all([p[i] == rry[i] for i in range(0, len(p))]):
            return True
    return False


class Block:

    def __init__(self, pos, size, faces, prog):
        # t0 = perf_counter()
        self.pos = pos
        self.size = size
        self.prog = prog
        self.calcs = len(faces[0])
        self.points = {}
        self.edges = []
        self.sides = []
        self.faces = []
        self.children = []
        self.parent_faces = faces
        self.flag = True

        if size >= 16:
            temp = 0
            for i in range(0, len(self.prog)):
                temp += self.prog[i] * 1 / (8 ** i)
            print(temp, self.prog, len(faces[0]))

        if self.size == 16:
            exact_pos = [math.floor((pos[0] + (model_scale / 2 - center)) / 16),
                         math.floor((pos[1] + (model_scale / 2 - center)) / 16),
                         math.floor((pos[2] + (model_scale / 2 - center)) / 16)]
            if "{}-{}-{}.mb3d".format(exact_pos[0], exact_pos[1], exact_pos[2]) in onlyfiles:
                self.flag = False

        if self.size <= 16:
            self.cube = np.chararray((int(self.size), int(self.size), int(self.size)), itemsize=16)
            self.cube[:] = "0"

        if self.flag:
            self.check_faces()
            self.subdivide()

    def check_faces(self):
        for x in range(-1, 2, 2):
            for y in range(-1, 2, 2):
                for z in range(-1, 2, 2):
                    start = np.array([self.pos[0] + x * self.size / 2, self.pos[1] + y * self.size / 2, self.pos[2] + z * self.size / 2])
                    if x < 0:
                        self.edges.append((start, np.array([self.size, 0, 0])))
                    if y < 0:
                        self.edges.append((start, np.array([0, self.size, 0])))
                    if z < 0:
                        self.edges.append((start, np.array([0, 0, self.size])))

        p1 = np.array([self.pos[0] - self.size / 2, self.pos[1] - self.size / 2, self.pos[2] - self.size / 2])

        p2 = np.array([self.pos[0] + self.size / 2, self.pos[1] - self.size / 2, self.pos[2] - self.size / 2])
        p3 = np.array([self.pos[0] - self.size / 2, self.pos[1] + self.size / 2, self.pos[2] - self.size / 2])
        p4 = np.array([self.pos[0] - self.size / 2, self.pos[1] - self.size / 2, self.pos[2] + self.size / 2])

        p5 = np.array([self.pos[0] - self.size / 2, self.pos[1] + self.size / 2, self.pos[2] + self.size / 2])
        p6 = np.array([self.pos[0] + self.size / 2, self.pos[1] - self.size / 2, self.pos[2] + self.size / 2])
        p7 = np.array([self.pos[0] + self.size / 2, self.pos[1] + self.size / 2, self.pos[2] - self.size / 2])

        p8 = np.array([self.pos[0] + self.size / 2, self.pos[1] + self.size / 2, self.pos[2] + self.size / 2])

        self.sides.append(CubeSide(p1, p2, p6, p4))
        self.sides.append(CubeSide(p1, p3, p7, p2))
        self.sides.append(CubeSide(p1, p3, p5, p4))

        self.sides.append(CubeSide(p8, p7, p3, p5))
        self.sides.append(CubeSide(p8, p6, p2, p7))
        self.sides.append(CubeSide(p8, p6, p4, p5))

        point_count = np.zeros((len(self.parent_faces[0])))
        points = np.zeros((len(self.parent_faces[0]), 2))

        truth = np.logical_and(self.parent_faces[0][:, 0, 0] >= self.pos[0] - self.size / 2,
                               self.parent_faces[0][:, 0, 0] <= self.pos[0] + self.size / 2)
        truth = np.logical_and(self.parent_faces[0][:, 0, 1] >= self.pos[1] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 0, 1] <= self.pos[1] + self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 0, 2] >= self.pos[2] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 0, 2] <= self.pos[2] + self.size / 2, truth)

        points[truth] = np.floor(np.einsum('ij,ijk->ik', self.parent_faces[0][:, 0], self.parent_faces[5][:])[truth])
        point_count[truth] += 1
        # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
        #    print("debug")

        truth = np.logical_and(self.parent_faces[0][:, 1, 0] >= self.pos[0] - self.size / 2,
                               self.parent_faces[0][:, 1, 0] <= self.pos[0] + self.size / 2)
        truth = np.logical_and(self.parent_faces[0][:, 1, 1] >= self.pos[1] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 1, 1] <= self.pos[1] + self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 1, 2] >= self.pos[2] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 1, 2] <= self.pos[2] + self.size / 2, truth)

        points[truth] = np.floor(np.einsum('ij,ijk->ik', self.parent_faces[0][:, 1], self.parent_faces[5][:])[truth])
        point_count[truth] += 1
        # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
        #    print("debug")

        truth = np.logical_and(self.parent_faces[0][:, 2, 0] >= self.pos[0] - self.size / 2,
                               self.parent_faces[0][:, 2, 0] <= self.pos[0] + self.size / 2)
        truth = np.logical_and(self.parent_faces[0][:, 2, 1] >= self.pos[1] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 2, 1] <= self.pos[1] + self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 2, 2] >= self.pos[2] - self.size / 2, truth)
        truth = np.logical_and(self.parent_faces[0][:, 2, 2] <= self.pos[2] + self.size / 2, truth)

        points[truth] = np.floor(np.einsum('ij,ijk->ik', self.parent_faces[0][:, 1], self.parent_faces[5][:])[truth])
        point_count[truth] += 1
        # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
        #    print("debug")

        for edge in self.edges:
            rel = -(np.sum(self.parent_faces[2][:] * edge[0], axis=1) + self.parent_faces[3][:]) / np.sum(self.parent_faces[2][:] * edge[1], axis=1)
            truth = np.logical_and(np.isfinite(rel), rel < 1)
            truth = np.logical_and(truth, rel > 0)

            p = np.asarray([edge[0]] * len(rel)) + np.asarray([edge[1]] * len(rel)) * np.transpose(
                np.asarray([rel] * 3))

            dis = np.sum(self.parent_faces[2][:] * p, axis=1) + self.parent_faces[3][:]
            point = p + self.parent_faces[2][:] * np.transpose(np.asarray([dis] * 3))
            edge_p1 = self.parent_faces[0][:, 0] - point
            edge_p2 = self.parent_faces[0][:, 1] - point
            edge_p3 = self.parent_faces[0][:, 2] - point
            alpha = np.linalg.norm(np.cross(edge_p1, edge_p2, axis=1), axis=1) / self.parent_faces[4][:]
            beta = np.linalg.norm(np.cross(edge_p2, edge_p3, axis=1), axis=1) / self.parent_faces[4][:]
            gamma = np.linalg.norm(np.cross(edge_p3, edge_p1, axis=1), axis=1) / self.parent_faces[4][:]

            perc = alpha + beta + gamma
            truth = np.logical_and(truth, perc <= 1.0000000000000018)
            truth = np.logical_and(truth, np.linalg.norm(edge_p1, axis=1) != 0.0)
            truth = np.logical_and(truth, np.linalg.norm(edge_p2, axis=1) != 0.0)
            truth = np.logical_and(truth, np.linalg.norm(edge_p2, axis=1) != 0.0)

            points[truth] = np.floor(np.einsum('ij,ijk->ik', point[truth], self.parent_faces[5][truth]))
            point_count[truth] += 1
            # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
            #    print("debug")

        for side in self.sides:
            side.normal = np.asarray([side.normal] * len(self.parent_faces[0]))
            side.d = np.asarray([side.d] * len(self.parent_faces[0]))
            side.area = np.asarray([side.area] * len(self.parent_faces[0]))
            side.p1 = np.asarray([side.p1] * len(self.parent_faces[0]))
            side.p2 = np.asarray([side.p2] * len(self.parent_faces[0]))
            side.p3 = np.asarray([side.p3] * len(self.parent_faces[0]))
            side.p4 = np.asarray([side.p4] * len(self.parent_faces[0]))

            # ab

            rel = -(np.sum(side.normal * self.parent_faces[0][:, 0], axis=1) + side.d) / np.sum(side.normal
                                                                                    * (self.parent_faces[0][:, 1]
                                                                                       - self.parent_faces[0][:, 0]), axis=1)
            truth = np.logical_and(np.isfinite(rel), rel < 1)
            truth = np.logical_and(truth, rel > 0)

            p = self.parent_faces[0][:, 0] + (self.parent_faces[0][:, 1] - self.parent_faces[0][:, 0]) * np.transpose(np.asarray([rel] * 3))

            dis = np.sum(side.normal * p, axis=1) + side.d
            point = p + side.normal * np.transpose(np.asarray([dis] * 3))
            side_p1 = side.p1 - point
            side_p2 = side.p2 - point
            side_p3 = side.p3 - point
            side_p4 = side.p4 - point
            alpha = np.linalg.norm(np.cross(side_p1, side_p2, axis=1), axis=1) / side.area
            beta = np.linalg.norm(np.cross(side_p2, side_p3, axis=1), axis=1) / side.area
            gamma = np.linalg.norm(np.cross(side_p3, side_p4, axis=1), axis=1) / side.area
            delta = np.linalg.norm(np.cross(side_p4, side_p1, axis=1), axis=1) / side.area

            perc = alpha + beta + gamma + delta
            truth = np.logical_and(truth, perc <= 1.0000000000000018)

            points[truth] = np.floor(np.einsum('ij,ijk->ik', point[truth], self.parent_faces[5][truth]))
            point_count[truth] += 1
            # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
            #    print("debug")

            # ac

            rel = -(np.sum(side.normal * self.parent_faces[0][:, 0], axis=1) + side.d) / np.sum(side.normal
                                                                                    * (self.parent_faces[0][:, 2]
                                                                                       - self.parent_faces[0][:, 0]), axis=1)
            truth = np.logical_and(np.isfinite(rel), rel < 1)
            truth = np.logical_and(truth, rel > 0)

            p = self.parent_faces[0][:, 0] + (self.parent_faces[0][:, 2] - self.parent_faces[0][:, 0]) * np.transpose(np.asarray([rel] * 3))

            dis = np.sum(side.normal * p, axis=1) + side.d
            point = p + side.normal * np.transpose(np.asarray([dis] * 3))
            side_p1 = side.p1 - point
            side_p2 = side.p2 - point
            side_p3 = side.p3 - point
            side_p4 = side.p4 - point
            alpha = np.linalg.norm(np.cross(side_p1, side_p2, axis=1), axis=1) / side.area
            beta = np.linalg.norm(np.cross(side_p2, side_p3, axis=1), axis=1) / side.area
            gamma = np.linalg.norm(np.cross(side_p3, side_p4, axis=1), axis=1) / side.area
            delta = np.linalg.norm(np.cross(side_p4, side_p1, axis=1), axis=1) / side.area

            perc = alpha + beta + gamma + delta
            truth = np.logical_and(truth, perc <= 1.0000000000000018)

            points[truth] = np.floor(np.einsum('ij,ijk->ik', point[truth], self.parent_faces[5][truth]))
            point_count[truth] += 1
            # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
            #    print("debug")

            # bc

            rel = -(np.sum(side.normal * self.parent_faces[0][:, 1], axis=1) + side.d) / np.sum(side.normal
                                                                                    * (self.parent_faces[0][:, 2]
                                                                                       - self.parent_faces[0][:, 1]), axis=1)
            truth = np.logical_and(np.isfinite(rel), rel < 1)
            truth = np.logical_and(truth, rel > 0)

            p = self.parent_faces[0][:, 1] + (self.parent_faces[0][:, 2] - self.parent_faces[0][:, 1]) * np.transpose(np.asarray([rel] * 3))

            dis = np.sum(side.normal * p, axis=1) + side.d
            point = p + side.normal * np.transpose(np.asarray([dis] * 3))
            side_p1 = side.p1 - point
            side_p2 = side.p2 - point
            side_p3 = side.p3 - point
            side_p4 = side.p4 - point
            alpha = np.linalg.norm(np.cross(side_p1, side_p2, axis=1), axis=1) / side.area
            beta = np.linalg.norm(np.cross(side_p2, side_p3, axis=1), axis=1) / side.area
            gamma = np.linalg.norm(np.cross(side_p3, side_p4, axis=1), axis=1) / side.area
            delta = np.linalg.norm(np.cross(side_p4, side_p1, axis=1), axis=1) / side.area

            perc = alpha + beta + gamma + delta
            truth = np.logical_and(truth, perc <= 1.0000000000000018)

            points[truth] = np.floor(np.einsum('ij,ijk->ik', point[truth], self.parent_faces[5][truth]))
            point_count[truth] += 1
            # if len(self.prog) >= 2 and self.prog[-1] == 2.0:
            #    print("debug")

        # if any(point_count == 1) or any(point_count == 2):
        #    print(np.where(point_count == 1))
        #    print(np.where(point_count == 2))
        #    print("debug")
        truth = point_count >= 3
        self.faces = [self.parent_faces[0][truth],
                      self.parent_faces[1][truth],
                      self.parent_faces[2][truth],
                      self.parent_faces[3][truth],
                      self.parent_faces[4][truth],
                      self.parent_faces[5][truth]]

        if any(truth) and self.size == 1:
            self.cube[0, 0, 0] = texture(points[truth])

        # print(f"Took: {perf_counter() - t0} [s]")

    def subdivide(self):
        if self.size > 1:
            if len(self.faces[0]) > 0:
                for x in np.arange(0, 1, 0.5):
                    for y in np.arange(0, 1, 0.5):
                        for z in np.arange(0, 1, 0.5):
                            newprog = deepcopy(self.prog)
                            newprog.append(8 * x + 4 * y + 2 * z)
                            new_pos = np.array(
                                [self.pos[0] + (-1 / 4 + x) * self.size, self.pos[1] + (-1 / 4 + y) * self.size,
                                 self.pos[2] + (-1 / 4 + z) * self.size])
                            self.children.append(Block(new_pos, self.size / 2, self.faces, newprog))
                            self.calcs += self.children[-1].calcs

                if size <= 16:
                    for c in self.children:
                        rel_pos = [0, 0, 0]
                        if self.pos[0] < c.pos[0]:
                            rel_pos[0] = self.size / 2
                        if self.pos[1] < c.pos[1]:
                            rel_pos[1] = self.size / 2
                        if self.pos[2] < c.pos[2]:
                            rel_pos[2] = self.size / 2
                        self.cube[int(rel_pos[0]):int(rel_pos[0] + self.size / 2),
                        int(rel_pos[1]):int(rel_pos[1] + self.size / 2),
                        int(rel_pos[2]):int(rel_pos[2] + self.size / 2)] = c.cube
                if size == 16 and np.any(self.cube != "0"):
                    export(self.cube, pos)

                self.children.clear()


class Mesh:

    def __init__(self, name):
        self.name = name
        self.vertex = [np.array([0, 0, 0])]
        self.texture = [np.array([0, 0])]
        self.face_vertex = []
        self.face_texture = []
        self.face_normal = None
        self.face_area = None
        self.face_d = None
        self.face_p_to_t = None


class CubeSide:

    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.ab = self.p2 - self.p1
        self.ac = self.p3 - self.p1
        self.bc = self.p3 - self.p2
        self.normal = np.cross(self.ab, self.ac)
        self.area = np.linalg.norm(self.normal) * 2
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.d = -self.normal[0] * self.p1[0] - self.normal[1] * self.p1[1] - self.normal[2] * self.p1[2]


start_time = time.clock()

dat = open("mesh.obj", "r")
img = Image.open("res.png", "r")
data32 = np.asarray(img, dtype="int32")
img = Image.open("res.png", "r")
data33 = np.asarray(img, dtype="int32")
alpha = Image.open("alpha.png", "r")
alpha = np.asarray(alpha, dtype="int32")
path = os.path.dirname(os.path.abspath(__file__))
onlyfiles = [f for f in os.listdir(path + "\\build") if os.path.isfile(os.path.join(path + "\\build", f))]
pinklist = []
pink = [255, 0, 255]
#build_size = 2048  # * 16
scale = ["max_extend", 1024]  # 1 * 16]
#block_size = 2 ** math.ceil(math.log(build_size, 2) + 1)
#print(block_size)

file = []
for line in dat:
    file.append(line)
dat.close()

meshes = {}
num = -1
for line in file:
    if line[0:2] == "g ":
        if num >= 0:
            meshes[num].vertex = np.asarray(meshes[num].vertex)
            meshes[num].texture = np.asarray(meshes[num].texture)
        num += 1
        meshes[num] = Mesh(line[2:-1])
    elif line[0:2] == "v ":
        temp = line[2:-1].split(" ")
        coord = np.array([float(temp[0]), float(temp[2]) * 1.5, float(temp[1])])
        meshes[num].vertex.append(np.asarray(coord[:]))
    elif line[0:2] == "vt":
        temp = line[3:-1].split(" ")
        coord = np.array([math.floor(float(temp[0]) * len(data32[0])), math.floor(float(temp[1]) * len(data32))])
        if coord[0] == len(data33[0]):
            coord[0] -= 1
        if coord[1] == len(data33):
            coord[1] -= 1
        data33[coord[1]][coord[0]][0] = 0
        data33[coord[1]][coord[0]][1] = 255
        data33[coord[1]][coord[0]][2] = 0
        meshes[num].texture.append(np.asarray(coord[:]))
    elif line[0:2] == "f ":
        temp = line[2:-1].split(" ")
        if len(temp) > 3:
            print("error please triangulate")
        for i in range(0, len(temp)):
            temp[i] = temp[i].split("/")
            try:
                temp[i] = np.array([int(temp[i][0]), int(temp[i][1])])
            except:
                temp[i] = np.array([int(temp[i][0]), 0])
        temp = np.array([temp[0], temp[1], temp[2]])
        if temp[0][0] == temp[1][0] or temp[0][0] == temp[2][0] or temp[2][0] == temp[1][0]:
            print("error please degenerate unnecessary faces")
        meshes[num].face_vertex.append(np.transpose(temp)[0])
        meshes[num].face_texture.append(np.transpose(temp)[1])

meshes[0].vertex = np.asarray(meshes[0].vertex)
meshes[0].texture = np.asarray(meshes[0].texture)
meshes[0].face_vertex = meshes[0].vertex[np.asarray(meshes[0].face_vertex)]
meshes[0].face_texture = meshes[0].texture[np.asarray(meshes[0].face_texture)]

x = [np.min(np.transpose(meshes[0].vertex)[0]), np.max(np.transpose(meshes[0].vertex)[0])]
y = [np.min(np.transpose(meshes[0].vertex)[1]), np.max(np.transpose(meshes[0].vertex)[1])]
z = [np.min(np.transpose(meshes[0].vertex)[2]), np.max(np.transpose(meshes[0].vertex)[2])]
print(x, y, z)

if scale[0] == "max":
    scale = scale[1] / max(abs(x[0]), x[1], abs(y[0]), y[1], abs(z[0]), z[1])
elif scale[0] == "x":
    scale = scale[1] / max(abs(x[0]), x[1])
elif scale[0] == "y":
    scale = scale[1] / max(abs(y[0]), y[1])
elif scale[0] == "z":
    scale = scale[1] / max(abs(z[0]), z[1])
elif scale[0] == "max_extend":
    scale = scale[1] / max(x[1] - x[0], y[1] - y[0], z[1] - z[0])
elif scale[0] == "x_extend":
    scale = scale[1] / (x[1] - x[0])
elif scale[0] == "y_extend":
    scale = scale[1] / (y[1] - y[0])
elif scale[0] == "z_extend":
    scale = scale[1] / (z[1] - z[0])
elif scale[0] == "value":
    scale = scale[1]
print(scale)
meshes[0].face_vertex = meshes[0].face_vertex * scale
meshes[0].vertex = meshes[0].vertex * scale
x = [np.min(np.transpose(meshes[0].vertex)[0]), np.max(np.transpose(meshes[0].vertex)[0])]
y = [np.min(np.transpose(meshes[0].vertex)[1]), np.max(np.transpose(meshes[0].vertex)[1])]
z = [np.min(np.transpose(meshes[0].vertex)[2]), np.max(np.transpose(meshes[0].vertex)[2])]

meshes[0].face_normal = np.cross(meshes[0].face_vertex[:, 1] - meshes[0].face_vertex[:, 0],
                                 meshes[0].face_vertex[:, 2] - meshes[0].face_vertex[:, 0], axis=1)
meshes[0].face_area = np.linalg.norm(meshes[0].face_normal, axis=1)
meshes[0].face_normal = meshes[0].face_normal / np.transpose(np.asarray([meshes[0].face_area] * 3))
meshes[0].face_d = - np.sum(meshes[0].face_normal[:] * meshes[0].face_vertex[:, 1], axis=1)
temp1 = np.linalg.solve(meshes[0].face_vertex, np.transpose(meshes[0].face_texture, (0, 2, 1))[:, 0, :])
temp2 = np.linalg.solve(meshes[0].face_vertex, np.transpose(meshes[0].face_texture, (0, 2, 1))[:, 1, :])
meshes[0].face_p_to_t = np.transpose(np.asarray([temp1, temp2]), (1, 2, 0))
model_scale = 2 ** np.ceil(np.log2(np.max(meshes[0].face_vertex) - np.min(meshes[0].face_vertex)))
center = 2 ** np.ceil(np.log2((np.max(meshes[0].face_vertex) + np.min(meshes[0].face_vertex))/2))
print(x, y, z)
test = Block(np.array([center, center, center]), model_scale, [meshes[0].face_vertex,
                                                               meshes[0].face_texture,
                                                               meshes[0].face_normal,
                                                               meshes[0].face_d,
                                                               meshes[0].face_area,
                                                               meshes[0].face_p_to_t], [0])
print(test.calcs)


path = os.path.dirname(os.path.abspath(__file__))

name_delta = [maxsize, maxsize, maxsize]
for file in os.listdir(path + "\\build"):
    if file[-5:] == ".mb3d":
        temp = file[:-5].split("-")
        for i in range(0, len(temp)):
            name_delta[i] = min(name_delta[i], int(temp[i]))

for file in os.listdir(path + "\\build"):
    if file[-5:] == ".mb3d":
        temp = file[:-5].split("-")
        for i in range(0, len(temp)):
            temp[i] = str(int(temp[i]) - name_delta[i])
        temp = temp[0] + "-" + temp[1] + "-" + temp[2] + ".mb3d"
        os.rename(path + "\\build\\" + file, path + "\\build\\" + temp)

np.clip(data33, 0, 255, out=data33)
data_u8 = data33.astype('uint8')
outimg = Image.fromarray(data_u8, "RGB")
outimg.save("test_2.png")
delta = time.clock() - start_time
