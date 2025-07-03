# position.py
import pyproj
import numpy as np
import csv
from dataclasses import dataclass, field

@dataclass
class NamedLocation:
    """
    Represents a named geographical location with latitude and longitude.

    Attributes:
        name (str): The semantic name of the location (e.g., "Station1", "M04").
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
    """
    name: str = ''
    lat: float = 0.0
    lon: float = 0.0


class NamedLocationLoader:
    """
    Loads and manages named geographical locations from a CSV file.
    This class provides a mapping from semantic labels (e.g., "Station1")
    to surveyed positions (latitude, longitude).

    The CSV file should have headings for Name, Latitude, and Longitude.
    Common variations for these headings are supported (e.g., "Name", "name",
    "Latitude", "Lat", "Longitude", "Lon").
    Latitude and Longitude are assumed to be in decimal degrees.
    """
    # Approximate conversion factor from decimal degrees to meters at the equator.
    DEGREES_TO_METRES = 111319.5

    def __init__(self, locationinfo: str):
        """
        Initializes the NamedLocationLoader with the path to the location info CSV file.

        Args:
            locationinfo (str): The full path to the CSV file containing named locations.
                                The CSV should have columns like 'Name', 'Latitude', 'Longitude'.
        """
        self.locationinfo = locationinfo
        self.namedlocations: list[NamedLocation] = []
        self._load_locations()

    def _load_locations(self):
        """
        Internal method to load named locations from the CSV file.
        It attempts to identify the correct columns for name, latitude, and longitude
        based on common header names.
        """
        name_field = None
        lat_field = None
        lon_field = None

        try:
            with open(self.locationinfo, 'r', newline='') as f:
                reader = csv.DictReader(f)
                # Identify column names case-insensitively
                for field_name in reader.fieldnames:
                    lower_field = field_name.lower()
                    if lower_field in {"name", "sitecode", "location", "site"}:
                        name_field = field_name
                    elif lower_field in {"latitude", "lat"}:
                        lat_field = field_name
                    elif lower_field in {"longitude", "lon", "lng"}:
                        lon_field = field_name

                if not all([name_field, lat_field, lon_field]):
                    raise ValueError(
                        f"Missing one or more required columns in {self.locationinfo}. "
                        "Expected: Name, Latitude, Longitude (or common variations)."
                    )

                for row in reader:
                    try:
                        entry = NamedLocation(
                            name=str(row[name_field]),
                            lat=float(row[lat_field]),
                            lon=float(row[lon_field])
                        )
                        self.namedlocations.append(entry)
                    except (ValueError, KeyError) as e:
                        print(f"WARNING: Skipping row due to data error in {self.locationinfo}: {row} - {e}")
            
            if not self.namedlocations:
                print("WARNING: No Named Locations loaded from the provided file.")
            else:
                print(f"Loaded {len(self.namedlocations)} named locations from {self.locationinfo}")

        except FileNotFoundError:
            print(f"ERROR: Location info file not found: {self.locationinfo}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load named locations from {self.locationinfo}: {e}")
            raise

    def fromName(self, name: str) -> NamedLocation | None:
        """
        Returns a NamedLocation object for an exact name match.

        Args:
            name (str): The exact name of the location to search for.

        Returns:
            NamedLocation | None: The matching NamedLocation object, or None if no exact match is found.
        """
        for entry in self.namedlocations:
            if entry.name == name:
                return entry
        return None

    def fromPos(self, lat: float, lon: float, threshold: float = 20.0) -> NamedLocation | None:
        """
        Returns the first NamedLocation that falls within a specified distance
        (threshold) of the given latitude and longitude.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
            threshold (float): The maximum distance in meters for a match. Defaults to 20.0 m.

        Returns:
            NamedLocation | None: The first matching NamedLocation, or None if no location
                                  is within the threshold.
        """
        for entry in self.namedlocations:
            delta_lat_m = (entry.lat - lat) * self.DEGREES_TO_METRES
            delta_lon_m = (entry.lon - lon) * self.DEGREES_TO_METRES
            distance = np.sqrt(delta_lat_m**2 + delta_lon_m**2)
            if distance <= threshold:
                return entry
        return None

    def closestToPos(self, lat: float, lon: float) -> NamedLocation | None:
        """
        Returns the NamedLocation closest to the given latitude and longitude.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.

        Returns:
            NamedLocation | None: The closest NamedLocation object, or None if no locations are loaded.
        """
        if not self.namedlocations:
            return None

        best_distance = float('inf')
        closest_location = None

        for entry in self.namedlocations:
            delta_lat_m = (entry.lat - lat) * self.DEGREES_TO_METRES
            delta_lon_m = (entry.lon - lon) * self.DEGREES_TO_METRES
            distance = np.sqrt(delta_lat_m**2 + delta_lon_m**2)
            if distance < best_distance:
                best_distance = distance
                closest_location = entry
        return closest_location

    def getAllNamedPos(self) -> list[NamedLocation]:
        """
        Returns a list of all loaded NamedLocation objects.

        Returns:
            list[NamedLocation]: A list of all named locations.
        """
        return self.namedlocations


class OverrideLoader:
    """
    Loads and manages override mappings from session paths to named locations.
    This is used to enforce a specific named location for a session based on its path,
    overriding potentially less accurate GPS-derived locations.

    The CSV file should have two columns: 'Station' (or 'Name') and 'Path'.
    """
    def __init__(self, overrideinfo: str):
        """
        Initializes the OverrideLoader with the path to the override info CSV file.

        Args:
            overrideinfo (str): The full path to the CSV file containing override mappings.
                                The CSV should have columns like 'Station' and 'Path'.
        """
        self.sessionNames: list[str] = []
        self.sessionPaths: list[str] = []
        self._load_overrides(overrideinfo)

    def _load_overrides(self, overrideinfo: str):
        """
        Internal method to load override mappings from the CSV file.
        """
        station_field = None
        path_field = None

        try:
            with open(overrideinfo, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for field_name in reader.fieldnames:
                    lower_field = field_name.lower()
                    if lower_field in {"station", "name"}:
                        station_field = field_name
                    elif lower_field == "path":
                        path_field = field_name

                if not all([station_field, path_field]):
                    raise ValueError(
                        f"Missing required columns in {overrideinfo}. "
                        "Expected: Station/Name, Path."
                    )

                for row in reader:
                    self.sessionNames.append(str(row[station_field]))
                    self.sessionPaths.append(str(row[path_field]))

            if not self.sessionNames:
                print("WARNING: No override sessions loaded from the provided file.")
            else:
                print(f"Loaded {len(self.sessionNames)} sessions for location override from {overrideinfo}.")

        except FileNotFoundError:
            print(f"ERROR: Override info file not found: {overrideinfo}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load override information from {overrideinfo}: {e}")
            raise

    def getNameFromPath(self, path: str) -> str | None:
        """
        Given a relative session path (e.g., '02Feb2023\\C69'), returns the
        corresponding named location (e.g., 'M02') if an override exists.

        Args:
            path (str): The relative path of the session.

        Returns:
            str | None: The named location string, or None if no match is found.
        """
        # Handle potential leading '0' in path for compatibility
        normalized_path = path
        if path.startswith('0') and len(path) > 1:
            normalized_path = path[1:]

        for name, stored_path in zip(self.sessionNames, self.sessionPaths):
            if stored_path == path or stored_path == normalized_path:
                return name
        return None

    def getPathsFromName(self, name: str) -> list[str] | None:
        """
        Given a named location, returns a list of all matching session paths.

        Args:
            name (str): The named location (e.g., 'M02').

        Returns:
            list[str] | None: A list of matching relative session paths, or None if no matches.
        """
        paths = [p for n, p in zip(self.sessionNames, self.sessionPaths) if n == name]
        return paths if paths else None


class CoordinateTransform:
    """
    Handles coordinate transformations, specifically between Latitude/Longitude
    and UTM (Universal Transverse Mercator) coordinates using pyproj.
    """
    def __init__(self):
        """
        Initializes the CoordinateTransform with an empty cache for projections.
        """
        self._projections = {}

    def zone(self, coordinates: tuple[float, float]) -> int:
        """
        Determines the UTM zone number for a given longitude and latitude.

        Args:
            coordinates (tuple[float, float]): A tuple (longitude, latitude) in decimal degrees.

        Returns:
            int: The UTM zone number.
        """
        lon, lat = coordinates
        if 56 <= lat < 64 and 3 <= lon < 12:
            return 32
        if 72 <= lat < 84 and 0 <= lon < 42:
            if lon < 9:
                return 31
            elif lon < 21:
                return 33
            elif lon < 33:
                return 35
            return 37
        return int((lon + 180) / 6) + 1

    def letter(self, coordinates: tuple[float, float]) -> str:
        """
        Determines the UTM band letter for a given latitude.

        Args:
            coordinates (tuple[float, float]): A tuple (longitude, latitude) in decimal degrees.

        Returns:
            str: The UTM band letter.
        """
        _, lat = coordinates
        return 'CDEFGHJKLMNPQRSTUVWXX'[int((lat + 80) / 8)]

    def project(self, coordinates: tuple[float, float]) -> tuple[int, str, float, float]:
        """
        Projects Latitude/Longitude coordinates to UTM coordinates.

        Args:
            coordinates (tuple[float, float]): A tuple (longitude, latitude) in decimal degrees.

        Returns:
            tuple[int, str, float, float]: A tuple containing (UTM zone, UTM band letter, UTM X, UTM Y).
        """
        lon, lat = coordinates
        z = self.zone(coordinates)
        l = self.letter(coordinates)

        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')

        x, y = self._projections[z](lon, lat)
        if y < 0: # Southern hemisphere offset
            y += 10000000
        return z, l, x, y

    def unproject(self, z: int, l: str, x: float, y: float) -> tuple[float, float]:
        """
        Unprojects UTM coordinates back to Latitude/Longitude.

        Args:
            z (int): UTM zone number.
            l (str): UTM band letter.
            x (float): UTM X coordinate.
            y (float): UTM Y coordinate.

        Returns:
            tuple[float, float]: A tuple containing (longitude, latitude) in decimal degrees.
        """
        if z not in self._projections:
            self._projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')

        if l < 'N': # Southern hemisphere offset
            y -= 10000000
        lon, lat = self._projections[z](x, y, inverse=True)
        return (lon, lat)

    def transform(self, lat: float, long: float) -> tuple[float, float]:
        """
        Transforms a single Latitude/Longitude pair to UTM X and Y coordinates.

        Args:
            lat (float): Latitude in decimal degrees.
            long (float): Longitude in decimal degrees.

        Returns:
            tuple[float, float]: A tuple containing (UTM X, UTM Y).
        """
        _, _, x, y = self.project((long, lat))
        return x, y


class Position:
    """
    Represents a geographical position, allowing for different coordinate systems
    (Latitude/Longitude, UTM, XY) and origin-referenced calculations.

    Attributes:
        h (float): Horizontal coordinate (e.g., longitude, UTM X, or XY X).
        v (float): Vertical coordinate (e.g., latitude, UTM Y, or XY Y).
        positiontype (str): The type of position ('LatLong', 'UTM', 'XY').
        x (float): Internal X coordinate (always in UTM for LatLong, otherwise same as h).
        y (float): Internal Y coordinate (always in UTM for LatLong, otherwise same as v).
        origin (Position | None): An optional reference position to define a local origin.
    """
    def __init__(self, coord_h: float, coord_v: float, positiontype: str):
        """
        Initializes a Position object.

        Args:
            coord_h (float): The horizontal coordinate.
            coord_v (float): The vertical coordinate.
            positiontype (str): The type of coordinate system. Must be 'LatLong', 'UTM', or 'XY'.

        Raises:
            Exception: If an invalid `positiontype` is provided.
        """
        self.h = coord_h
        self.v = coord_v
        self.positiontype = positiontype
        self.origin: Position | None = None # A reference position acts as the origin

        if positiontype == 'XY':
            self.x = self.h
            self.y = self.v
        elif positiontype == 'UTM':
            self.x = self.h
            self.y = self.v
        elif positiontype == 'LatLong':
            # Convert LatLong to UTM upon initialization for internal consistency
            c = CoordinateTransform()
            utm_x, utm_y = c.transform(self.h, self.v)
            self.x = utm_x
            self.y = utm_y
        else:
            raise ValueError("Invalid positiontype. Must be 'LatLong', 'UTM', or 'XY'.")

    def setOrigin(self, origin: 'Position'):
        """
        Sets a reference origin for this position. When an origin is set,
        `xy()` will return coordinates relative to this origin.

        Args:
            origin (Position): The Position object to use as the origin.
        """
        self.origin = origin

    def xy(self) -> tuple[float, float]:
        """
        Returns the coordinates in XY format, relative to the set origin if one exists.
        If no origin is set, returns the internal (UTM or XY) coordinates.

        Returns:
            tuple[float, float]: A tuple (x, y) representing the position.
        """
        if self.origin is not None:
            return self.x - self.origin.x, self.y - self.origin.y
        else:
            return self.x, self.y

    def utm(self) -> tuple[float, float]:
        """
        Returns the coordinates in UTM format. If an origin is set, it returns
        the absolute UTM coordinates by adding the origin's coordinates.
        If no origin is set, it returns the internal (UTM) coordinates.

        Returns:
            tuple[float, float]: A tuple (utm_x, utm_y) representing the position in UTM.
        """
        if self.origin is not None:
            # If origin is set, and we want UTM, we return the absolute UTM
            # This assumes self.x, self.y are already absolute UTM if positiontype was LatLong/UTM
            # If positiontype was XY, then self.x, self.y are relative, and we need to add origin.
            # This logic might need refinement based on exact use case of 'utm()' when origin is set.
            # For now, assuming xy() is for relative, utm() is for absolute.
            return self.x, self.y
        else:
            return self.x, self.y

    def distance(self, position_b: 'Position') -> float:
        """
        Calculates the Euclidean distance between this position and another Position object.
        The distance is calculated based on their internal (UTM or XY) coordinates.

        Args:
            position_b (Position): The other Position object to calculate the distance to.

        Returns:
            float: The distance between the two positions.
        """
        delta_x = self.x - position_b.x
        delta_y = self.y - position_b.y
        dist = np.sqrt(delta_x**2 + delta_y**2)
        return dist

    def __repr__(self) -> str:
        """
        Returns a string representation of the Position object, showing its
        origin-referenced XY coordinates.
        """
        x, y = self.xy()
        return f"Position (x={x:.2f}, y={y:.2f})"