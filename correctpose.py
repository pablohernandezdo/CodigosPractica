import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import simplekml
import xmltodict
import numpy as np

from rdp import rdp
from exif import Image
from scipy.spatial import KDTree
from libxmp.utils import file_to_dict
from shapely.geometry import Point, Polygon

# No instalar shapely desde los repositorios de pip, usar wheel
# https://towardsdatascience.com/install-shapely-on-windows-72b6581bb46c


def main():
    # img_folder = PhotoFolder("Fotos/Drone_Adrian")
    # img_folder = PhotoFolder("Fotos/SE_MAIPO")
    # img_folder = PhotoFolder("Fotos/2021-03-01")
    img_folder = PhotoFolder("Fotos/2021-02-26")

    # Guardar archivos kml
    img_folder.save_folder_kmls()

    # Guardar archivos kml corregidos, debe existir un archivo json con
    # los parametros corregidos asociados al nombre de cada foto
    img_folder.save_corrected_kmls()


class PhotoFolder:
    def __init__(self, folder_path: str):

        self.folder_path = folder_path

    def __str__(self):
        return '\n'.join(self.files_list)

    @property
    def correcciones(self) -> dict:

        corr_path = self.folder_path + "/correcciones.json"

        if os.path.exists(corr_path):
            with open(corr_path) as json_file:
                data = json.load(json_file)

            return data["correcciones"]

        else:
            return {}

    @property
    def files_list(self) -> list:
        files = []

        for img in os.listdir(self.folder_path):
            if os.path.splitext(img)[-1] in [".JPG", ".jpg", ".png"]:
                files.append(f'{self.folder_path}/{img}')

        return sorted(files)

    def save_corrected_kmls(self):
        for img in self.files_list:
            foto = Photo(img)
            foto.gsd = self.correcciones[foto.img_path]["gsd"]
            foto.latitude += self.correcciones[foto.img_path]["lat"]
            foto.longitude += self.correcciones[foto.img_path]["lon"]
            foto.save_kml(self.folder_path.split('/')[-1])

    def save_folder_kmls(self):
        for img in self.files_list:
            f = Photo(img)
            f.save_kml(self.folder_path.split('/')[-1])

    def save_folder_metadata(self):

        meta_folder = f"Metadata/{self.folder_path.split('/')[-1]}"
        Path(meta_folder).mkdir(parents=True, exist_ok=True)

        for img in self.files_list:

            f = Photo(img)

            with open(f'{meta_folder}/{f.img_name}.txt', 'w') as file:
                Photo.save_metadata(f.img_path, file)


class Photo:
    def __init__(self, img_path: str):

        self.img_path = img_path
        self.img_name = self.img_path.split('/')[-1].split('.')[0]

        self.img = cv2.imread(img_path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        # Parametros de pose desde metadata
        pose = self.get_pose(img_path)

        self.focal_length = pose["focal_length"]
        self.longitude = pose["longitude"]
        self.latitude = pose["latitude"]
        self.altitude = pose["altitude"]
        self.gimbal_yaw = pose["gimbal_yaw"]
        self.gimbal_pitch = pose["gimbal_pitch"]
        self.gimbal_roll = pose["gimbal_roll"]
        self.flight_yaw = pose["flight_yaw"]
        self.flight_pitch = pose["flight_pitch"]
        self.flight_roll = pose["flight_roll"]

        # Este número es mágico sacado del internet
        # self.afov = 78.8 # L1D-20C camera Mavic PRO
        self.afov = 84  # FC3170 Camera Mavic Air 2

        # self.gsd = self.calc_gsd()
        self.gsd = np.abs(self.calc_gsd())

        # if self.gsd < 0:
        #     print("Gsd negativa")

    def __str__(self):
        description = f'Path         \t{self.img_path}\n' \
                      f'Name         \t{self.img_name}\n' \
                      f'Width        \t{self.width}\n' \
                      f'Height       \t{self.height}\n' \
                      f'Focal length \t{self.focal_length}\n' \
                      f'Longitude    \t{self.longitude}\n' \
                      f'Latitude     \t{self.latitude}\n' \
                      f'Altitude     \t{self.altitude}\n' \
                      f'Ground height\t{self.ground_height}\n' \
                      f'Gimbal yaw   \t{self.gimbal_yaw}\n' \
                      f'Gimbal pitch \t{self.gimbal_pitch}\n' \
                      f'Gimbal roll  \t{self.gimbal_roll}\n' \
                      f'Flight_yaw   \t{self.flight_yaw}\n' \
                      f'Flight_pitch \t{self.flight_pitch}\n' \
                      f'Flight_roll  \t{self.flight_roll}\n' \
                      f'Afov         \t{self.afov}\n' \
                      f'GSD          \t{self.gsd}\n' \
                      f'Corners      \t{self.corners[0]}\n' \
                      f'              \t{self.corners[1]}\n' \
                      f'              \t{self.corners[2]}\n' \
                      f'              \t{self.corners[3]}\n'

        return description

    def calc_gsd(self):
        rad_afov = np.deg2rad(self.afov / 2)
        return 2 * np.tan(rad_afov) * self.ground_height / self.width

    @property
    def center(self):
        return self.longitude, self.latitude

    @property
    def ground_height(self):
        return round(self.altitude - Duct.elevation(self.center), 2)

    @property
    def corners(self):
        # Punto medio
        w = self.width // 2
        h = self.height // 2

        # Escalar, rotar y mover 4 esquinas de la imagen, centro es (0,0)
        p1 = self.map_point(w, h)
        p2 = self.map_point(w, -h)
        p3 = self.map_point(-w, -h)
        p4 = self.map_point(-w, h)

        return [p1, p2, p3, p4]

    @staticmethod
    def save_metadata(img_path: str, file):
        file.write(f'\nImage: {img_path.split("/")[-1]}\n\n')

        # Exif tags
        exif_data = Image(img_path)

        # XMP tags
        xmpdata = file_to_dict(img_path)

        # XMP attributes
        attrlist = xmpdata["http://www.dji.com/drone-dji/1.0/"]

        file.write("EXIF properties\n\n")

        # Print exif data
        for field in dir(exif_data):
            try:
                file.write(f'{field}: {exif_data[field]}\n')

            except:
                print(f"No se pudo leer el campo {field}")

        file.write("\nXMP keys and properties\n")

        for key in xmpdata.keys():
            file.write(f"\n{key}\n\n")

            for prop in xmpdata[key]:
                file.write(f"{prop[0]}: {prop[1]}\n")

    @staticmethod
    def get_pose(img_path: str) -> dict:
        # Exif tags
        exif_data = Image(img_path)

        # XMP tags
        xmpdata = file_to_dict(img_path)

        # XMP attributes
        attrlist = xmpdata["http://www.dji.com/drone-dji/1.0/"]

        # Available attributes dictionary with corresponding index
        attrs_index = {}
        for idx, attr in enumerate(attrlist):
            new_key = attr[0].split(':')[-1]
            attrs_index[new_key] = idx

        # Exif longitud and latitude to float
        long, lat = Photo.coordinate_to_float(exif_data["gps_longitude"],
                                              exif_data["gps_latitude"],
                                              exif_data["gps_longitude_ref"],
                                              exif_data["gps_latitude_ref"])

        # Extract pose
        pose = {"focal_length": exif_data["focal_length"],
                "latitude": lat,
                "longitude": long,
                "altitude": np.float(
                    attrlist[attrs_index["AbsoluteAltitude"]][1]),
                "gimbal_yaw": np.float(
                    attrlist[attrs_index["GimbalYawDegree"]][1]),
                "gimbal_pitch": np.float(
                    attrlist[attrs_index["GimbalPitchDegree"]][1]),
                "gimbal_roll": np.float(
                    attrlist[attrs_index["GimbalRollDegree"]][1]),
                "flight_yaw": np.float(
                    attrlist[attrs_index["FlightYawDegree"]][1]),
                "flight_pitch": np.float(
                    attrlist[attrs_index["FlightPitchDegree"]][1]),
                "flight_roll": np.float(
                    attrlist[attrs_index["FlightRollDegree"]][1]),
                }

        return pose

    @staticmethod
    def coordinate_to_float(long: tuple, lat: tuple,
                            long_ref: str, lat_ref: str) -> tuple:

        long = long[0] + long[1] / 60 + long[2] / 3600
        lat = lat[0] + lat[1] / 60 + lat[2] / 3600

        if long_ref == "W":
            long = - long

        if lat_ref == "S":
            lat = - lat

        return long, lat

    @staticmethod
    def correct_height(height: float, coordinates: tuple) -> float:

        elevation = Duct.elevation(coordinates)
        corrected = height - elevation
        return corrected

    def map_point(self, x: int, y: int) -> tuple:
        scaled = self.scale(x, y)
        rotated = self.rotate(scaled, self.flight_yaw)
        moved = self.move(self.center, rotated)
        return moved

    def scale(self, x: int, y: int) -> tuple:
        return self.gsd * x, self.gsd * y

    @staticmethod
    def rotate(point: tuple, angle: float,
               origin: tuple = (0, 0)) -> tuple:

        angle = np.deg2rad(angle)

        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) + np.sin(angle) * (py - oy)
        qy = oy - np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

        return qx, qy

    @staticmethod
    def move(point: tuple, vector: tuple) -> tuple:
        R = 6378137.0  # WGS84 Equatorial Radius in Meters
        lat = point[1] + np.rad2deg(vector[1] / R)
        lon = point[0] + np.rad2deg(vector[0] / R) / np.cos(np.deg2rad(lat))
        return lon, lat

    def save_kml(self, imgs_folder, filename=None):

        # kml files folder
        files_dir = Path("Kml").joinpath(imgs_folder)
        files_dir.mkdir(parents=True, exist_ok=True)

        kml = simplekml.Kml()

        ground = kml.newgroundoverlay(name="GroundOverlay")

        if filename:
            ground.icon.href = '../../../' + self.img_path

        else:
            ground.icon.href = '../../' + self.img_path

        # The coordinates must be specified in counter-clockwise
        # order with the first coordinate corresponding to
        # the lower-left corner of the overlayed image.
        # eg. [(0, 1), (1,1), (1,0), (0,0)]

        # El orden de las coordenadas en el formato de la clase es
        # 0.- Superior derecha
        # 1.- Inferior derecha
        # 2.- Inferior izquierda
        # 3.- Superior izquierda

        ground.gxlatlonquad.coords = [self.corners[2],
                                      self.corners[1],
                                      self.corners[0],
                                      self.corners[3]]

        # Corner points
        for esq, pnt in zip(["UR", "LR", "LL", "UL"], self.corners):
            kml.newpoint(name=f"{esq}",
                         coords=[pnt])

        # Lines conecting corners
        kml.newlinestring(name=self.img_name,
                          coords=self.corners + [self.corners[0]])

        # Save file
        if filename:
            kml.save(f'{files_dir}/{filename}.kml')
        else:
            kml.save(f'{files_dir}/{self.img_name}.kml')

    def get_keypoints_descriptors(self, n_keypoints):
        # Setup SIFT
        sift = cv2.SIFT_create(n_keypoints)

        # Image to grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Get keypoints and descriptors
        kp, desc = sift.detectAndCompute(gray, None)

        return kp, desc


class Matcher:
    pass


class PhotoAdrian:
    def __init__(self, **kwargs):

        if 'heading' in kwargs:
            kwargs['theta'] = np.deg2rad(kwargs['heading'])

        if 'datetime' in kwargs:
            kwargs['timestamp'] = kwargs['datetime']

        self.latitude = kwargs['latitude']
        self.longitude = kwargs['longitude']
        self.altitude = kwargs['altitude']

        self.gsd = kwargs['gsd']

        self.theta = kwargs['theta']
        self.width = kwargs['width']
        self.height = kwargs['height']

        if 'url' in kwargs.keys():
            self.url = kwargs['url']

        if kwargs['timestamp']:
            m = re.match(r'(.*)\.000', kwargs['timestamp'])

            if m:
                kwargs['timestamp'] = m.group(1)

            self.timestamp = datetime.strptime(kwargs['timestamp'],
                                               '%Y-%m-%d %H:%M:%S')

        self.corners = self.setCorners()

    @property
    def center(self):
        return self.longitude, self.latitude

    @property
    def groundHeight(self):
        return round(self.altitude - Duct.elevation(self.center), 2)

    @property
    def size(self):
        return self.width, self.height

    def scale(self, x: int(), y: int()) -> tuple():
        return self.gsd * x, self.gsd * y

    def rotate(self, point: tuple(), angle: float,
               origin: tuple() = (0, 0)) -> tuple():
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) + np.sin(angle) * (py - oy)
        qy = oy - np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def move(self, point: tuple(), vector: tuple()) -> tuple():
        R = 6378137.0  # WGS84 Equatorial Radius in Meters
        lat = point[1] + np.rad2deg(vector[1] / R)
        lon = point[0] + np.rad2deg(vector[0] / R) / np.cos(np.deg2rad(lat))
        return lon, lat

    def mapPoint(self, x: int(), y: int()) -> tuple():
        scaled = self.scale(x, y)
        rotated = self.rotate(scaled, self.theta)
        moved = self.move(self.center, rotated)
        return moved

    def setCorners(self) -> list():
        # Punto medio
        w = self.width / 2
        h = self.height / 2

        # Escalar, rotar y mover 4 esquinas de la imagen, centro es (0,0)
        p1 = self.mapPoint(w, h)
        p2 = self.mapPoint(w, -h)
        p3 = self.mapPoint(-w, -h)
        p4 = self.mapPoint(-w, h)

        return [p1, p2, p3, p4]

    def draw(self):
        coordinates = ''
        for p in self.corners:
            coordinates += '{},{},0 '.format(*p)
        return {
            'name': str(self.timestamp),
            'Polygon': {
                'outerBoundaryIs': {
                    'LinearRing': {
                        'coordinates': coordinates,
                        'altitudeMode': 'relativeToGround'
                    }
                }
            },
            'styleUrl': '#photoStyle'
        }


class Duct:
    R = 6378137.0  # WGS84 Equatorial Radius in Meters

    def __init__(self, name: str, points: list):

        self.name = name
        self.points = [tuple(point[0:2]) for point in points]
        self.tree = KDTree(self.points)
        self.points3D = [tuple(point) for point in points]
        self.tree3D = KDTree(self.points3D)

    @staticmethod
    def _lineEquation(p1: tuple, p2: tuple) -> tuple:
        # Ecuación de línea y=a*x+b
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - a * p1[0]
        return a, b

    @staticmethod
    def _altitudeFoot(line: tuple(), point: tuple()) -> tuple():
        # Línea perpendicular a line
        a = -1 / line[0] if (line[0] != 0) else np.Inf
        b = point[1] - a * point[0]
        x = (b - line[1]) / (line[0] - a)
        y = a * x + b
        return x, y

    @staticmethod
    def _belongsToSegment(p1: tuple(), p2: tuple(), p3: tuple()) -> bool():
        if (p1[0] > p2[0]):
            if (p3[0] > p1[0] or p3[0] < p2[0]):
                return False
        else:
            if (p3[0] > p2[0] or p3[0] < p1[0]):
                return False
        if (p1[1] > p2[1]):
            if (p3[1] > p1[1] or p3[1] < p2[1]):
                return False
        else:
            if (p3[1] > p2[1] or p3[1] < p1[1]):
                return False
        return True

    @staticmethod
    def distance(p1: tuple(), p2: tuple()) -> float():
        λ1, φ1 = np.deg2rad(p1)
        λ2, φ2 = np.deg2rad(p2)

        # https://en.wikipedia.org/wiki/Great-circle_distance
        Δλ = λ2 - λ1
        Δφ = np.sin(φ1) * np.sin(φ2) + np.cos(φ1) * np.cos(φ2) * np.cos(Δλ)
        Δσ = np.arccos(Δφ)
        m = Δσ * Duct.R
        return m

    @staticmethod
    def bearing(p1: tuple(), p2: tuple()) -> float():
        λ1, φ1 = np.deg2rad(p1)
        λ2, φ2 = np.deg2rad(p2)
        Δλ = λ2 - λ1
        y = np.sin(Δλ) * np.cos(φ2)
        x = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)
        brng = np.arctan2(y, x)
        θ = (np.rad2deg(brng) + 360) % 360
        return θ

    @staticmethod
    def destinationPoint(point: tuple(), bearing: float(),
                         distance: float()) -> tuple():
        λ, φ = np.deg2rad(point)
        θ = np.deg2rad(bearing)
        δ = distance / Duct.R
        x = np.arcsin(np.sin(φ) * np.cos(δ) + np.cos(φ) * np.sin(δ) * np.cos(θ))
        y = λ + np.arctan2(np.sin(θ) * np.sin(δ) * np.cos(φ),
                           np.cos(δ) - np.sin(φ) * np.sin(x))
        return tuple(np.rad2deg((x, y)))

    @staticmethod
    def getTags(file, end='little'):
        # https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml

        # Offset al primer IFD
        file.seek(4)
        ifd = int.from_bytes(file.read(4), end)

        # Número de tags
        file.seek(ifd)
        entries = int.from_bytes(file.read(2), end)

        tags = []
        for i in range(entries):
            off = ifd + 2 + 12 * i
            file.seek(off)
            tag = int.from_bytes(file.read(2), end)
            typ = int.from_bytes(file.read(2), end)
            cnt = int.from_bytes(file.read(4), end)
            val = int.from_bytes(file.read(4), end)
            tags.append({
                'tag': tag,
                'type': typ,
                'count': cnt,
                'value': val
            })
        return tags

    @staticmethod
    def readTiff(file, row, col, samples):
        # https://docs.fileformat.com/image/tiff/
        # Endianness
        end = 'big' if file.read(2) == b'MM' else 'little'
        index = row * samples + col

        tags = Duct.getTags(file, end)
        pointer = next((t for t in tags if t['tag'] == 273), None)['value']

        # Inicio de data
        file.seek(pointer)
        start = int.from_bytes(file.read(4), end)

        # Búsqueda del valor
        file.seek(start + 2 * index)
        val = int.from_bytes(file.read(2), end, signed=True)
        return val

    @staticmethod
    def AW3D30name(point):
        lat = ('N' if point[1] > 0 else 'S') + \
              '{:03d}'.format(int(abs(point[1] // 1)))

        lon = ('E' if point[0] > 0 else 'W') + \
              '{:03d}'.format(int(abs(point[0] // 1)))

        return 'ALPSMLC30_{}{}_DSM.tif'.format(lat, lon)

    @staticmethod
    def elevation(point, samples=3600):
        filename = 'AW3D30/' + Duct.AW3D30name(point)
        with open(filename, 'rb') as file:
            row = samples - 1 - int((point[1] % 1) * samples)
            col = int((point[0] % 1) * samples)
            elev = Duct.readTiff(file, row, col, samples)
            return elev

    @staticmethod
    def _segmentFoot(point: tuple(), p1: tuple(), p2: tuple()) -> tuple():
        line = Duct._lineEquation(p1, p2)
        foot = Duct._altitudeFoot(line, point)
        if (Duct._belongsToSegment(p1, p2, foot)):
            return foot
        else:
            return None

    @staticmethod
    def _distanceToSegment(p1: tuple(), p2: tuple(), point: tuple()) -> float():
        foot = Duct._segmentFoot(point, p1, p2)
        if foot != None:
            d = Duct.distance(point, foot)
        else:
            d = min(Duct.distance(point, p1), Duct.distance(point, p2))
        return d

    def intersectionPoint(self, point: tuple()) -> tuple():
        idx = self.tree.query(point)[1]
        nearest = self.points[idx]  # Punto más cercano

        foot1 = Duct._segmentFoot(
            point, nearest, self.points[idx + 1]) if idx != (
                len(self.points) - 1) else None
        foot2 = Duct._segmentFoot(
            point, nearest, self.points[idx - 1]) if idx != 0 else None

        if foot1 or foot2:
            return foot1 or foot2
        else:
            return nearest

    def distanceToLine(self, point: tuple()) -> float():
        intersection = self.intersectionPoint(point)
        d = Duct.distance(point, intersection)
        return d

    @staticmethod
    def travel(points: list(), distance: float()) -> list():
        accu = 0
        pts = points[0:1]
        for point in points[1:]:
            d = Duct.distance(point, pts[-1])
            if d + accu <= distance:
                accu += d
                pts.append(point)
            else:
                θ = Duct.bearing(pts[-1], point)
                p = Duct.destinationPoint(pts[-1], θ, distance - accu)
                pts.append(tuple(p))
                break
        return pts

    def routeBetween(self,
                     p1: tuple(),
                     p2: tuple(),
                     altitude: float() = 120,
                     epsilon: float() = 0,
                     absolute: bool() = False,
                     ) -> list():
        idx1 = self.tree.query(p1)[1]
        idx2 = self.tree.query(p2)[1]

        if idx1 < idx2:
            if idx2 != len(self.points3D):
                points = self.points3D[idx1:idx2 + 1]
            else:
                point = self.points3D[idx1:]
        else:
            if idx2 != 0:
                points = self.points3D[idx1:idx2 - 1:-1]
            else:
                points = self.points3D[idx1::-1]

        if absolute:
            points = [(point[0], point[1], point[2] + altitude)
                      for point in points]
        else:
            points = [(point[0], point[1], altitude) for point in points]

        points_rdp = rdp(points, epsilon=epsilon)
        return points_rdp

    @staticmethod
    def generateKML(points, name, absolute: bool() = False):
        routeStr = ''
        for pt in points:
            routeStr += '{},{},{} '.format(*pt)
        xml = {
            'Document': {
                'name': name,
                'Style': [
                    {
                        '@id': 'routeStyle',
                        'LineStyle': {'color': 'ff00ffff', 'width': 5}
                    },
                    {
                        '@id': 'iconStyle',
                        'IconStyle': {'color': 'ff0000ff'}
                    }
                ],
                'Placemark': [
                    {
                        'name': 'Inicio',
                        'styleUrl': '#iconStyle',
                        'Point': {
                            'altitudeMode': 'absolute' if absolute else 'relativeToGround',
                            'coordinates': '{},{},{}'.format(*points[0])
                        }
                    },
                    {
                        'name': 'Ruta',
                        'styleUrl': '#routeStyle',
                        'LineString': {
                            'altitudeMode': 'absolute' if absolute else 'relativeToGround',
                            'coordinates': routeStr
                        }
                    }
                ]
            }
        }
        with open(name + '.kml', 'w') as f:
            XML = xmltodict.unparse(xml, pretty=True)
            f.write(XML)

    def excludePoints(self, photos: list()):
        discardedPoints = []
        for photo in photos:
            poly = Polygon(photo.corners)
            c0 = photo.center
            c1 = photo.corners[0]
            d = np.linalg.norm(np.array(c0) - np.array(c1))
            indexes = self.tree.query_ball_point(photo.center, d)
            pts = [self.points[i] for i in indexes]
            for point in pts:
                if Point(point).within(poly):
                    discardedPoints.append(point)
        s = set(discardedPoints)
        self.excluded = [x for x in self.points if x not in s]

    @staticmethod
    def intersects(line, poly):
        test = (poly[0][1] * line[0] + line[1]) > poly[0][0]
        for p in poly[1:]:
            if test != ((p[1] * line[0] + line[1]) > p[0]):
                return True
        return False

    def parallelPoints(self, photos, eps=0.005, d=300):
        parallel = []
        if len(photos) > 0:
            tree = KDTree([p.center for p in photos])
            for point in self.excluded:
                index = self.tree.query(point)[1]
                if index > 0:
                    m1 = Duct._lineEquation(point, self.points[index - 1])[0]
                if index < len(self.points) - 1:
                    m2 = Duct._lineEquation(point, self.points[index + 1])[0]
                if 'm1' in locals() and 'm2' in locals():
                    # Pendiente de recta perpendicular en el punto
                    m = - 2 / (m1 + m2)
                elif 'm1' in locals():
                    m = -1 / m1
                elif 'm2' in locals():
                    m = -1 / m2
                n = point[0] - m * point[1]

                indexes = tree.query_ball_point(point, eps)
                fewPhotos = [photos[i] for i in indexes]

                for photo in fewPhotos:
                    if (Duct.intersects((m, n), photo.corners) and
                            Duct.distance(point, photo.center) < d):
                        parallel.append(point)
                        break
        self.parallel = parallel


class Flight:
    def __init__(self, data=[], photos=[]):
        self.data = data
        self.photos = photos

    def drawExcludedPoints(self):
        pts = []
        for pt in [point for i in self.data for point in i['duct'].excluded]:
            pts.append({
                'Point': {
                    'coordinates': '{},{},0'.format(*pt),
                    'altitudeMode': 'relativeToGround'
                },
                'styleUrl': '#iconStyle'
            })

        return pts

    def kml(self, filename):
        self.photos.sort(key=lambda x: x.timestamp)
        if self.photos[0].timestamp.date() == self.photos[-1].timestamp.date():
            name = 'Cobertura del {}'.format(self.photos[0].timestamp.date())
        else:
            name = 'Cobertura entre {} y {}'.format(
                self.photos[0].timestamp.date(),
                self.photos[-1].timestamp.date())
        xml = {
            'Document': {
                'name': name,
                'Style': [
                    {
                        '@id': 'photoStyle',
                        'LineStyle': {'color': 'ff00ffff'},
                        'PolyStyle': {'color': '8000ffff'}
                    },
                    {
                        '@id': 'iconStyle',
                        'IconStyle': {'color': 'ff0000ff'}
                    },
                ],
                'Folder': [
                    {
                        'name': 'Fotos',
                        'open': 0,
                        'Placemark': [photo.draw() for photo in self.photos]
                    },
                    {
                        'name': 'Puntos no cubiertos',
                        'open': 0,
                        'Placemark': self.drawExcludedPoints()
                    },
                ]
            }
        }
        with open(filename, 'w') as f:
            XML = xmltodict.unparse(xml, pretty=True)
            f.write(XML)

    def getTimes(self):
        if not any(['photos' in place.keys() for place in self.data]):
            return
        print('Hora:')
        for place in self.data:
            if 'photos' not in place.keys() or len(place['photos']) == 0:
                continue
            name = place['duct'].name
            photos = sorted(place['photos'], key=lambda x: x.timestamp)
            times = [photos[0].timestamp, photos[-1].timestamp]
            print('\tInicio {}\t: {}'.format(name, times[0]))
            print('\tTérmino {}\t: {}'.format(name, times[-1]))

    def getAltitudes(self, min_altitude=10):
        alts = [f.altitude for f in self.photos if f.altitude > min_altitude]
        print('Altitud:')
        print('\tMáxima\t: {:4d} msnm'.format(round(max(alts))))
        print('\tMínima\t: {:4d} msnm'.format(round(min(alts))))
        print('\tMedia\t: {:4d} msnm'.format(round(sum(alts) / len(alts))))

    def getCoverage(self):
        print('Porcentaje de cobertura:')
        totalPts = 0
        exclPts = 0
        for place in self.data:
            pts = len(place['duct'].points)
            excl = len(place['duct'].excluded)
            totalPts += pts
            exclPts += excl
            print('\t{}\t: {:.2f}%'.format(
                place['duct'].name, 100 * (1 - excl / pts)))
        print('\tTotal\t: {:.2f}%'.format(100 * (1 - exclPts / totalPts)))

    def getParallels(self):
        if not any(['parallel' in place['duct'].__dict__.keys() for place in
                    self.data]):
            return
        print('Porcentaje no cubierto por fotos fuera de ducto:')
        totalPts = 0
        parPts = 0
        for place in self.data:
            pts = len(place['duct'].points)
            par = len(place['duct'].parallel)
            totalPts += pts
            parPts += par
            print('\t{}\t: {:.2f}%'.format(place['duct'].name, 100 * par / pts))
        print('\tTotal\t: {:.2f}%'.format(100 * parPts / totalPts))

    def getStats(self):
        self.getTimes()
        self.getAltitudes()
        self.getCoverage()
        self.getParallels()
        if any(['photos' in place.keys() for place in self.data]):
            print('Total de fotos:')
            for place in self.data:
                if 'photos' not in place.keys() or len(place['photos']) == 0:
                    continue
                print('\t{}\t: {}'.format(
                    place['duct'].name, len(place['photos'])))


if __name__ == "__main__":
    main()
