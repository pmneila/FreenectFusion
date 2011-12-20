
import math
from math import cos, sin
from operator import add, mul

def cross(v1, v2):
    return (v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0])

def normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return tuple(map(lambda x: x/length, v))

class OrbitCamera(object):
    
    def __init__(self):
        self.center = (0.0,)*3
        self.radius = 1000.0
        self.longitude = 3*math.pi/2.0
        self.latitude = 0.0
    
    def rotate(self, dlongitude, dlatitude):
        self.longitude += dlongitude
        self.latitude += dlatitude
        if self.latitude >= math.pi/2.0:
            self.latitude = math.pi/2.0 - 1e-3
        if self.latitude <= -math.pi/2.0:
            self.latitude = -math.pi/2.0 + 1e-3
    
    def translate(self, dx, dy):
        # Get the perpendicular plane.
        v = self.get_vector()
        e1 = normalize(cross(v, (0,1,0)))
        e2 = normalize(cross(v, e1))
        e1 = map(lambda x: x*dx, e1)
        e2 = map(lambda x: x*dy, e2)
        displ = map(add, e1, e2)
        self.center = tuple(map(add, self.center, displ))
    
    def zoom(self, dy):
        step = self.radius/100
        if dy <= -100:
            dy = -99
        self.radius += step*dy
        if self.radius < 1.0:
            self.radius = 1.0
    
    def get_position(self):
        return tuple(map(add, self.center, self.get_vector()))
    
    def get_vector(self):
        return (cos(self.longitude)*cos(self.latitude)*self.radius,
                sin(self.latitude)*self.radius,
                sin(self.longitude)*cos(self.latitude)*self.radius)
    
    def get_glulookat_parameters(self):
        return self.position + self.center + (0,1,0)
    
    position = property(get_position)
