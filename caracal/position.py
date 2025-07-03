
import pyproj
import numpy as np

import csv
from dataclasses import dataclass,field
import numpy

@dataclass 
class NamedLocation:
    name: str = ''
    lat: float = 0.0
    lon: float = 0.0


class NamedLocationLoader:
    # Decimal degrees to metres conversion (approximate)
    DEGREES_TO_METRES = 111319.5
    def __init__(self,locationinfo):
        '''Initialize a location mapper instance.
        This provides mapping from semantic labels (e.g. "Station1") to
        surveyed positions (lat,lon). This allows semantic labels to be
        used as query strings. 
        
        Broadly the flow is:
        1. Semantic label (e.g. Station1) 
              V
              V
        2. Surveyed Location (e.g. x.xxx, y.yyyy)
              V
              V
        3. Location Measured by the Device (e.g. p.pppp, q.qqqq)
              V
              V
        4. Session ID/Device ID/Audio File
        
        This class takes care of stages 1 and 2
        
        Provide a locationinfo file which is a .csv with headings:

        Name, Latitude, Longitude

        For Name, the following will be tried:
        "Name", "name"

        For Latitude the following will be tried:
        "Latitude", "latitude", "Lat, "lat"

        Extra fields will be ignored
        
        Lat and Lon are assumed to be in decimal degrees
        '''
        self.locationinfo = locationinfo
        self.namedlocations = []
        namefield = None
        latfield = None
        lonfield = None
        with open(locationinfo) as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames:
                if field.lower() in {"name","sitecode","location","site"}:
                    namefield = field
                if field.lower() in {"latitude", "lat"}:
                    latfield = field
                if field.lower() in {"longitude", "lon", "lng"}:
                    lonfield = field
            # Print matching fields
            for row in reader:
                entry = NamedLocation(str(row[namefield]),
                                      float(row[latfield]),
                                      float(row[lonfield]))
                self.namedlocations.append(entry)
        if len(self.namedlocations) == 0:
            print("WARNING: No Named Locations loaded!")
        else:
            print("Loaded",len(self.namedlocations),"named locations")

    def fromName(self,name):
        '''Return a NamedLocation for the exact name match, else return None'''
        for e in self.namedlocations:
            if e.name == name:
                return e
        # Explicit none return for no match
        return None

    def fromPos(self,lat,lon,threshold=20.0):
        '''
        Return a NamedLocation for the given position, else return None. 

        This will return the first location that matches the threshold criteria

        Params:
        - lat is latitude in decimal degrees
        - lon is longitude in decimal degrees
        - Threshold is distance in [m]

        Returns:
        - NamedLocation
        '''
        for e in self.namedlocations:
            delta_lat = (e.lat-lat)*NamedLocationLoader.DEGREES_TO_METRES
            delta_lon = (e.lon-lon)*NamedLocationLoader.DEGREES_TO_METRES
            dist = numpy.sqrt(delta_lat**2+delta_lon**2)
            if dist <= threshold:
                return e
        # Explicit none return for no match
        return None
    
    def closestToPos(self,lat,lon):
        '''
        Return the closest NamedLocation for the given position.

        Params:
        - lat is latitude in decimal degrees
        - lon is longitude in decimal degrees
        - Threshold is distance in [m]

        Returns:
        - NamedLocation
        '''
        bestDist = 1e10
        bestN = None
        for e in self.namedlocations:
            delta_lat = (e.lat-lat)*NamedLocationLoader.DEGREES_TO_METRES
            delta_lon = (e.lon-lon)*NamedLocationLoader.DEGREES_TO_METRES
            dist = numpy.sqrt(delta_lat**2+delta_lon**2)
            if dist < bestDist:
                bestDist = dist
                bestN = e
        return bestN


    def getAllNamedPos(self):
        '''Return a list of NamedLocations'''
        return self.namedlocations
    


class OverrideLoader:

    def __init__(self,overrideinfo):
        '''Initialize an override instance

        This is a .csv file which has two columns:
        Station,Path
        M02,02Feb2023\C69
        ...

        This is used to map a particular session (by path) to a particular named location
        '''
        self.sessionNames = []
        self.sessionPaths = []
        with open(overrideinfo) as f:
            reader = csv.DictReader(f)
            for field in reader.fieldnames:
                if field.lower() == "station" or field.lower() == "name":
                    stationfield = field
                if field.lower() == "path":
                    pathfield = field
            for row in reader:
                self.sessionNames.append(row[stationfield])
                self.sessionPaths.append(row[pathfield])
            print("Loaded:",len(self.sessionNames),"sessions for location over-ride.")

    def getNameFromPath(self,path):
        ''' Supply a relative path e.g. 02Feb2023\C69 and return the named location e.g. 'M02'.
        Return None if no match'''
        for n,p in zip(self.sessionNames,self.sessionPaths):
            if (p == path) or (p == '0'+path):
                return n
        return None
        
    def getPathsFromName(self,name):
        ''' Supply a name, and get a list of matching paths back e.g. ['02Feb2023\C69','09Feb2023\C69']
        Returns None if no matches'''
        paths = []
        for n,p in zip(self.sessionNames,self.sessionPaths):
            if n == name:
                paths.append(p)
        if len(paths) == 0:
            return None
        return paths

class CoordinateTransform():
    
    def __init__(self):
        """ This class deals with coordinate transformations"""
        self._projections = {}

    def zone(self,coordinates):
        if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
            return 32
        if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
            if coordinates[0] < 9:
                return 31
            elif coordinates[0] < 21:
                return 33
            elif coordinates[0] < 33:
                return 35
            return 37
        return int((coordinates[0] + 180) / 6) + 1


    def letter(self,coordinates):
        return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


    def project(self,coordinates):
        z = self.zone(coordinates)
        l = self.letter(coordinates)
        #print(z,l)
        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        x, y = self._projections[z](coordinates[0], coordinates[1])
        if y < 0:
            y += 10000000
        return z, l, x, y


    def unproject(self,z, l, x, y):
        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
        if l < 'N':
            y -= 10000000
        lng, lat = self._projections[z](x, y, inverse=True)
        return (lng, lat)

    def transform(self,lat,long):
        z,l,x,y = self.project((long,lat))
        return x,y


class Position():
    def __init__(self,coord_h,coord_v,positiontype):
        """positiontype sets the type of position. Can be:
        - 'LatLong'
        - 'UTM'
        - 'XY'        """
        self.h = coord_h
        self.v = coord_v
        self.positiontype = positiontype
        # A reference position acts as the origin
        self.origin = None
        if positiontype == 'XY':
            self.x = self.h
            self.y = self.v
        elif positiontype  == 'UTM':
            self.x = self.h
            self.y = self.v
        elif positiontype == 'LatLong':
            c = CoordinateTransform()
            utm_x,utm_y = c.transform(self.h,self.v)
            self.x = utm_x
            self.y = utm_y
        else:
            raise Exception("Invalid positiontype")

    
    def setOrigin(self,origin):
        self.origin = origin
        
    def xy(self):
        """Return coordinates in origin referenced XY format"""
        if self.origin is not None:
            return self.x - self.origin.x,self.y - self.origin.y
        else:
            return self.x,self.y
        
    def utm(self):
        """Return coordinates in UTM format"""
        if self.origin is not None:
            return self.x,self.y
        else:
            return self.x+self.origin.x,self.y+self.origin.y
        
    def distance(self,position_b):
        """Distance to another position"""
        delta_x = self.x - position_b.x
        delta_y = self.y - position_b.y
        dist = np.sqrt(delta_x**2+delta_y**2)
        return dist
    
    def __repr__(self):
        x,y = self.xy()
        return "Position (" + str(x) + ":" + str(y) + ")" 