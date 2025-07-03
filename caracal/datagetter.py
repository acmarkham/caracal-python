# datagetter.py
import pickle
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo
import datetime
import os
import numpy as np # Use np for numpy operations for consistency
import soundfile as sf
import csv
import obspy # Import obspy

from .syslogparser import Identity, Stats, AudioFile, Header, Session, SyslogContainer
from .position import NamedLocation, NamedLocationLoader, OverrideLoader

# Container for multiple audio data returns
@dataclass
class CaracalMultipleAudioData:
    """
    A container for multiple CaracalAudioData instances, representing audio
    data from one or more stations or sessions for a given query.

    Attributes:
        stations (list[CaracalAudioData]): A list of CaracalAudioData objects.
    """
    stations: list['CaracalAudioData'] = field(default_factory=list)

    def saveWav(self, filename: str):
        """
        Saves all audio data from all stations in this container into a single
        multi-channel WAV file. If there's only one station, it saves a mono/quad WAV.
        If there are multiple stations, it attempts to concatenate them if they have
        the same sample rate and duration, or saves them individually if not.
        For simplicity, this implementation saves each station's audio as a separate
        file, named with an index.

        Args:
            filename (str): The base filename (without extension) for saving.
                            Individual files will be named like 'filename_0.wav', 'filename_1.wav', etc.
        """
        if not self.stations:
            print("No audio data to save.")
            return

        for i, caracal_audio in enumerate(self.stations):
            if caracal_audio.audioData is not None and caracal_audio.sampleRate > 0:
                output_filename = f"{filename}_{i}.wav"
                try:
                    sf.write(output_filename, caracal_audio.audioData, caracal_audio.sampleRate)
                    print(f"Saved audio for station {i} to {output_filename}")
                except Exception as e:
                    print(f"Error saving audio for station {i} to {output_filename}: {e}")
            else:
                print(f"No valid audio data for station {i} to save.")

    def to_obspy_stream(self) -> obspy.Stream:
        """
        Converts the CaracalMultipleAudioData instance into an ObsPy Stream object.
        Each CaracalAudioData instance will be converted into an ObsPy Trace,
        and these traces will be added to an ObsPy Stream.

        Returns:
            obspy.Stream: An ObsPy Stream object containing all converted traces.
        """
        stream = obspy.Stream()
        for caracal_audio in self.stations:
            if caracal_audio.audioData is not None and caracal_audio.sampleRate > 0:
                try:
                    # Ensure audioData is a 1D array for mono, or 2D for multi-channel
                    if caracal_audio.audioData.ndim > 1:
                        # If quad, Obspy Trace expects (samples, channels) but usually works with (samples,)
                        # For simplicity, let's assume the primary channel or flatten if needed.
                        # For quad, we might need to create multiple traces or handle multi-channel explicitly.
                        # Assuming the first channel for now if it's multi-dimensional.
                        data = caracal_audio.audioData[:, 0] if caracal_audio.audioData.shape[1] > 0 else caracal_audio.audioData.flatten()
                    else:
                        data = caracal_audio.audioData

                    # Create an ObsPy Trace
                    trace = obspy.Trace(data=data, header={
                        'sampling_rate': caracal_audio.sampleRate,
                        'starttime': obspy.UTCDateTime(caracal_audio.UTCstart),
                        'station': caracal_audio.stationName,
                        'network': 'XX', # Placeholder network code
                        'location': caracal_audio.locationMatch,
                        'channel': 'BHZ' # Placeholder channel code
                    })

                    # Add additional metadata if available
                    if caracal_audio.header.headerID.deviceID:
                        trace.stats.device_id = caracal_audio.header.headerID.deviceID
                    if caracal_audio.header.stats.median_GPS_lat and caracal_audio.header.stats.median_GPS_lon:
                        trace.stats.coordinates = {
                            'latitude': caracal_audio.header.stats.median_GPS_lat,
                            'longitude': caracal_audio.header.stats.median_GPS_lon
                        }
                    stream.append(trace)
                except Exception as e:
                    print(f"Error converting CaracalAudioData to ObsPy Trace for station {caracal_audio.stationName}: {e}")
            else:
                print(f"Skipping ObsPy conversion for station {caracal_audio.stationName}: No valid audio data or sample rate.")
        return stream


# Container for single station's data
@dataclass
class CaracalAudioData:
    """
    A container for audio data from a single station, along with associated metadata.

    Attributes:
        path (str): The relative path to the syslog file associated with this data.
        header (Header): Header information from the syslog.
        audioFile (AudioFile): Information about the specific audio file.
        UTCstart (int): UTC timestamp (seconds) of the start of the audio segment.
        UTCEnd (int): UTC timestamp (seconds) of the end of the audio segment.
        sampleRate (int): The sample rate of the audio data.
        audioData (np.ndarray): The actual audio data as a NumPy array.
        locationMatch (str): Describes how the location was matched (e.g., 'FromDevice', 'FromOverride', 'Unspecified').
        stationName (str): The name of the station, if identified.
    """
    path: str = ''
    header: Header = field(default_factory=Header)
    audioFile: AudioFile = field(default_factory=AudioFile)
    UTCstart: int = 0
    UTCEnd: int = 0
    sampleRate: int = 0
    audioData: np.ndarray | None = None
    locationMatch: str = 'Unspecified'
    stationName: str = 'Unknown'

    def saveWav(self, filename: str):
        """
        Saves the audio data from this single CaracalAudioData instance to a WAV file.

        Args:
            filename (str): The full path and filename for the output WAV file.
        """
        if self.audioData is not None and self.sampleRate > 0:
            try:
                sf.write(filename, self.audioData, self.sampleRate)
                print(f"Saved audio to {filename}")
            except Exception as e:
                print(f"Error saving audio to {filename}: {e}")
        else:
            print("No valid audio data or sample rate to save.")


class DataGetter:
    """
    Provides methods to retrieve audio data and associated metadata from
    a collection of parsed syslog files, based on various query parameters
    like device ID, geographical location, or named stations.
    """

    # Index of the mono channel in the wave file to use (if applicable)
    CH_MONO_IDX = 0
    # Decimal degrees to meters conversion (approximate, accurate at the equator)
    DEGREES_TO_METRES = 111319.5

    def __init__(self,
                 rootpath: str,
                 syslogdata: str,
                 locationinfo: str | None = None,
                 timezone: str = "UTC",
                 audio_mode: str = 'mono',
                 overrideinfo: str | None = None):
        """
        Initializes a DataGetter instance.

        Args:
            rootpath (str): The main, top-level path where all the data lives.
                            Typically, this will be the same path as used for InventoryBuilder.
            syslogdata (str): Path to a pickled file containing a list of SyslogContainer objects.
            locationinfo (str, optional): Path to a CSV file for semantic/surveyed locations.
                                          Used by NamedLocationLoader. Defaults to None.
            timezone (str): Timezone string (ZoneInfo code, e.g., "America/New_York").
                            Used to convert naive datetimes to aware timestamps. Defaults to "UTC".
            audio_mode (str): Specifies the audio channel mode. 'mono' for single channel,
                              'quad' for four channels. Defaults to 'mono'.
            overrideinfo (str, optional): Path to a CSV file for overriding semantic/surveyed locations.
                                          Used by OverrideLoader. Defaults to None.

        Raises:
            ValueError: If an invalid `audio_mode` is provided.
        """
        self.rootpath = rootpath
        self.syslogdata = syslogdata
        self.locationinfo = locationinfo
        self.loc: NamedLocationLoader | None = None
        self.override: OverrideLoader | None = None

        if audio_mode not in ['mono', 'quad']:
            raise ValueError("Audio mode must be 'mono' or 'quad'")
        self.audio_mode = audio_mode
        print(f"DataGetter: Audio Mode set to {self.audio_mode}")

        self.timezone_str = timezone
        self.local_timezone = ZoneInfo(timezone)
        print(f"DataGetter: Timezone set to: {self.local_timezone}")

        # Load the parsed syslogs from the pickled file
        try:
            with open(syslogdata, 'rb') as handle:
                self.sys: list[SyslogContainer] = pickle.load(handle)
                print(f"DataGetter: Loaded {len(self.sys)} syslog entries.")
        except FileNotFoundError:
            print(f"ERROR: Syslog data file not found at {syslogdata}. Please ensure it exists.")
            self.sys = []
        except Exception as e:
            print(f"ERROR: Failed to load syslog data from {syslogdata}: {e}")
            self.sys = []

        if locationinfo:
            try:
                self.loc = NamedLocationLoader(locationinfo)
            except Exception as e:
                print(f"ERROR: Could not initialize NamedLocationLoader with {locationinfo}: {e}")
        if overrideinfo:
            try:
                self.override = OverrideLoader(overrideinfo)
            except Exception as e:
                print(f"ERROR: Could not initialize OverrideLoader with {overrideinfo}: {e}")

    def __convert_datetime_to_utc_timestamp(self, dt: datetime.datetime) -> float:
        """
        Internal helper to convert a datetime object to a UTC Unix timestamp.
        If the datetime is naive, it's assumed to be in the local_timezone
        set during DataGetter initialization.

        Args:
            dt (datetime.datetime): The datetime object to convert.

        Returns:
            float: The UTC Unix timestamp.
        """
        # Check if timezone has been set. If not, assume that it is in the naive timezone and convert
        # to an aware date-time.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.local_timezone)
        # First change to UTC if necessary
        utc_dt = dt.astimezone(ZoneInfo("UTC"))
        # Then convert to unix timestamps
        utc = utc_dt.timestamp()
        return utc

    @staticmethod
    def __unpack(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal function to inflate a packed 2-channel (int32) wave file
        into four 16-bit channels with gain applied.
        Assumes input `data` is a NumPy array of shape (num_samples, 2) and dtype int32.

        Args:
            data (np.ndarray): Packed audio data as a NumPy array (num_samples, 2), dtype int32.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - new_data (np.ndarray): Unpacked 4-channel audio data (num_samples, 4), dtype int32.
                - gain_array (np.ndarray): Array of gain values for each sample.
        """
        num_samples = data.shape[0]
        new_data = np.empty((num_samples, 4), dtype=np.int32)
        gain_array = np.empty(num_samples, dtype=np.float32) # Use float32 for gain

        # Ensure data is int32 for bit manipulation
        data_int = data.astype(np.int32)

        for idx in np.arange(num_samples):
            d0 = data_int[idx, 0]
            d1 = data_int[idx, 1]

            # Extract channels and gain from packed 32-bit integers
            ch_a = (d0 & 0xFFFFFE00)
            ch_b = (d1 & 0xFFF00000) >> 20
            ch_c = (d1 & 0x000FFF00) >> 8
            ch_d = ((d1 & 0x000000FF) << 4) | ((d0 & 0x0F0) >> 4) # Corrected d0 & 0x0F to d0 & 0x0F0 >> 4
            gain = (d0 & 0x1F0) >> 4 # 5 bits for gain

            # Signed conversion for the mantissa (12-bit signed for B, C, D; 23-bit signed for A)
            ch_a = (ch_a ^ 0x80000000) - 0x80000000 if ch_a & 0x80000000 else ch_a # 32-bit signed
            ch_b = (ch_b ^ 0x800) - 0x800 if ch_b & 0x800 else ch_b # 12-bit signed
            ch_c = (ch_c ^ 0x800) - 0x800 if ch_c & 0x800 else ch_c # 12-bit signed
            ch_d = (ch_d ^ 0x800) - 0x800 if ch_d & 0x800 else ch_d # 12-bit signed

            # Apply exponent (gain) to channels B, C, D
            # Note: 2**gain can be large, ensure types handle it
            ch_b = ch_b * (2**gain)
            ch_c = ch_c * (2**gain)
            ch_d = ch_d * (2**gain)

            # ACM 16APR24 - update the unpacker to get the angles in order: mic_angle = [0,90,180,270]
            # Assuming original order was A, B, C, D and desired is A, B, D, C for 0, 90, 270, 180
            new_data[idx, :] = [ch_a, ch_b, ch_d, ch_c]
            gain_array[idx] = gain
        return new_data, gain_array

    @staticmethod
    def load_wav(filename: str, start_offset: float | None = None,
                 duration: float | None = None, audio_mode: str = 'mono') -> tuple[int, np.ndarray]:
        """
        Loads audio from a WAV file, supporting partial loading and different audio modes.
        This method uses the `soundfile` library for efficient seeking and reading.

        Args:
            filename (str): Full path to the .wav file to open.
            start_offset (float, optional): Offset in seconds (can be fractional)
                                            relative to the start timestamp of the file.
                                            Defaults to None (start from beginning).
            duration (float, optional): How many seconds of audio (can be fractional) to load.
                                        Required.
            audio_mode (str): 'mono' for single channel (CH_MONO_IDX), 'quad' for four channels.
                              Defaults to 'mono'.

        Returns:
            tuple[int, np.ndarray]: A tuple containing the sample rate (int) and
                                    the audio data as a NumPy array.

        Raises:
            ValueError: If `duration` is None or if `audio_mode` is invalid.
            Exception: If the file is not seekable or other soundfile errors occur.
        """
        if duration is None:
            raise ValueError("Need a duration to load audio.")

        try:
            with sf.SoundFile(filename, 'r') as track:
                if not track.seekable():
                    raise ValueError(f"Audio file '{filename}' is not seekable.")

                sr = track.samplerate
                start_frame = int((start_offset or 0) * sr)
                frames_to_read = int(sr * duration)

                track.seek(start_frame)

                if audio_mode == 'mono':
                    # Read all channels, then select the mono index
                    audio_section = track.read(frames_to_read, dtype='int16') # Read as int16, then convert if needed
                    # Ensure it's 2D if original is multi-channel, then squeeze
                    if audio_section.ndim > 1:
                        audio_data = np.squeeze(audio_section[:, DataGetter.CH_MONO_IDX])
                    else:
                        audio_data = audio_section
                    return sr, audio_data
                elif audio_mode == 'quad':
                    # Read raw packed data as int32
                    audio_section = track.read(frames_to_read, dtype='int32')
                    quad_audio, gain = DataGetter.__unpack(audio_section)
                    # Scale to float64 [-1,1] range if needed, or keep as int32
                    # For consistency with other audio libraries, often float is preferred.
                    # Max value for int32 is 2^31 - 1, so divide by 2^31
                    audio_scaled = quad_audio.astype(np.float64) / (2**31)
                    return sr, audio_scaled
                else:
                    raise ValueError("Audio mode must be 'mono' or 'quad'")
        except Exception as e:
            print(f"Failed to load WAV file '{filename}': {e}")
            raise

    @staticmethod
    def load_wav_legacy(filename: str, start_offset: float | None = None,
                        duration: float | None = None) -> tuple[int, np.ndarray] | None:
        """
        Loads (mono) audio using `scipy.io.wavfile`. This method is less efficient
        for large files or partial reads compared to `load_wav`.

        Args:
            filename (str): Full path to the .wav file to open.
            start_offset (float, optional): Offset in seconds (can be fractional)
                                            relative to the start timestamp of the file.
                                            Defaults to None (start from beginning).
            duration (float, optional): How many seconds of audio (can be fractional) to load.
                                        Defaults to None (read until end).

        Returns:
            tuple[int, np.ndarray] | None: A tuple containing the sample rate (int) and
                                           the audio data as a NumPy array, or None if loading fails.
        """
        try:
            samplerate, data = wavfile.read(filename)
        except Exception as e:
            print(f"Failed to load WAV file '{filename}' using legacy method: {e}")
            return None

        offset_idx = int((start_offset or 0) * samplerate)
        if duration is not None:
            end_idx = int(offset_idx + duration * samplerate)
            # Ensure slicing is within bounds
            end_idx = min(end_idx, data.shape[0])
            start_idx = min(offset_idx, data.shape[0])
            return samplerate, np.squeeze(data[start_idx:end_idx, DataGetter.CH_MONO_IDX])
        else:
            start_idx = min(offset_idx, data.shape[0])
            return samplerate, np.squeeze(data[start_idx:, DataGetter.CH_MONO_IDX])

    def get_audio_from_session(self, session: Session, start_timestamp: float, end_timestamp: float) -> CaracalMultipleAudioData:
        """
        Retrieves audio data for a specified time range from a given session.
        This method can span multiple hourly-aligned audio files within a session
        if the query time range crosses file boundaries.

        Args:
            session (Session): The session object from which to extract audio.
            start_timestamp (float): UTC Unix timestamp (seconds) for the start of the desired audio.
            end_timestamp (float): UTC Unix timestamp (seconds) for the end of the desired audio.

        Returns:
            CaracalMultipleAudioData: A container with the requested audio data.
                                      Will contain one or more CaracalAudioData instances,
                                      concatenated if they are contiguous and from the same station.
        """
        matching_audio_data_list: list[CaracalAudioData] = []
        
        # Filter audio files within the session that overlap with the query time range
        relevant_audio_files = []
        for audiofile in session.audioFiles:
            file_utc_start = audiofile.utcTime
            file_utc_end = audiofile.utcTime + audiofile.duration

            # Check for overlap: [start_timestamp, end_timestamp] overlaps with [file_utc_start, file_utc_end]
            if max(start_timestamp, file_utc_start) < min(end_timestamp, file_utc_end):
                relevant_audio_files.append(audiofile)
        
        # Sort relevant files by their UTC start time to ensure correct concatenation order
        relevant_audio_files.sort(key=lambda af: af.utcTime)

        # Process each relevant audio file
        for audiofile in relevant_audio_files:
            file_utc_start = audiofile.utcTime
            file_utc_end = audiofile.utcTime + audiofile.duration

            # Determine the actual segment to load from this specific file
            segment_start_utc = max(start_timestamp, file_utc_start)
            segment_end_utc = min(end_timestamp, file_utc_end)

            if segment_start_utc >= segment_end_utc:
                # No actual overlap for this file, or segment is zero length
                continue

            offset_in_file = segment_start_utc - file_utc_start
            duration_to_load = segment_end_utc - segment_start_utc

            full_filename = os.path.join(self.rootpath, session.path, audiofile.subpath)

            try:
                sr, aud = DataGetter.load_wav(full_filename,
                                              offset_in_file,
                                              duration_to_load,
                                              audio_mode=self.audio_mode)

                if aud is not None:
                    c = CaracalAudioData(
                        path=session.path,
                        header=session.header,
                        audioFile=audiofile,
                        UTCstart=int(segment_start_utc),
                        UTCEnd=int(segment_end_utc),
                        sampleRate=sr,
                        audioData=aud,
                        stationName=session.header.headerID.deviceID # Use device ID as station name by default
                    )
                    matching_audio_data_list.append(c)
            except Exception as e:
                print(f"Failed to get audio from {full_filename} for segment [{segment_start_utc}, {segment_end_utc}]: {e}")
        
        # If multiple audio segments are found, attempt to concatenate them if they are contiguous
        # and from the same logical "station" (e.g., same device ID, same sample rate).
        # For now, we'll return them as separate CaracalAudioData instances within the list.
        # A more advanced concatenation logic could be added here if needed,
        # but for simplicity, each loaded segment becomes a distinct CaracalAudioData.
        
        # If the user wants a single continuous audio stream, this is where concatenation would happen.
        # For now, we return a list of individual segments.
        # Example of simple concatenation if all segments are from the same station and contiguous:
        # if len(matching_audio_data_list) > 1:
        #     # Check for contiguity and same sample rate, then concatenate audioData
        #     # This requires careful handling of timestamps and potential gaps/overlaps.
        #     pass

        return CaracalMultipleAudioData(stations=matching_audio_data_list)


    def load_from_session(self, session: Session, start_time: float | datetime.datetime, end_time: float | datetime.datetime) -> CaracalMultipleAudioData:
        """
        Retrieves audio data for a specified time range from a given session.
        This method will now use `get_audio_from_session` which handles multi-file spans.

        Args:
            session (Session): The session object from which to extract audio.
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.

        Returns:
            CaracalMultipleAudioData: A container with the requested audio data.
                                      Will contain one or more CaracalAudioData instances.
        """
        if not isinstance(session, Session):
            print("ERR: Input is not a Session instance.")
            return CaracalMultipleAudioData(stations=[]) # Return empty container

        # Convert datetime objects to UTC timestamps if necessary
        if isinstance(start_time, datetime.datetime):
            start_time = self.__convert_datetime_to_utc_timestamp(start_time)
        if isinstance(end_time, datetime.datetime):
            end_time = self.__convert_datetime_to_utc_timestamp(end_time)

        # Use the updated get_audio_from_session which handles multi-file queries
        return self.get_audio_from_session(session, start_time, end_time)


    def load_from_device_id(self, devID: str, start_time: float | datetime.datetime, end_time: float | datetime.datetime) -> CaracalMultipleAudioData:
        """
        Loads audio data for a given device ID within a specified time range.
        This method can return multiple CaracalAudioData instances if the device
        recorded data across different sessions or files within the time range.

        Args:
            devID (str): The device ID to query.
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.

        Returns:
            CaracalMultipleAudioData: A container with all matching CaracalAudioData instances.
        """
        # Convert datetime objects to UTC timestamps if necessary
        if isinstance(start_time, datetime.datetime):
            start_time = self.__convert_datetime_to_utc_timestamp(start_time)
        if isinstance(end_time, datetime.datetime):
            end_time = self.__convert_datetime_to_utc_timestamp(end_time)

        all_matching_audio_data: list[CaracalAudioData] = []

        # Step 1: Scan through all syslog containers and sessions to find matches by deviceID
        for syslog_container in self.sys:
            for session in syslog_container.sessions:
                session_dev_id = session.header.headerID.deviceID
                if devID == session_dev_id:
                    # Found a session matching the device ID, now check time overlap
                    session_start_time = session.audioFiles[0].utcTime if session.audioFiles else float('inf')
                    session_end_time = session.audioFiles[-1].utcTime + session.audioFiles[-1].duration if session.audioFiles else float('-inf')

                    if max(start_time, session_start_time) < min(end_time, session_end_time):
                        # The session itself overlaps with the query time
                        # Call get_audio_from_session to extract relevant audio segments
                        # This will handle spanning multiple files within that session
                        multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                        all_matching_audio_data.extend(multi_audio_data.stations)
        
        return CaracalMultipleAudioData(stations=all_matching_audio_data)


    def load_from_latlon(self, lat: float, lon: float,
                         start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                         location_threshold: float = 20.0) -> CaracalMultipleAudioData:
        """
        Loads audio data from stations located near a specified latitude/longitude
        within a given time range. This method prioritizes device GPS locations,
        then falls back to override locations if available.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.
            location_threshold (float): Maximum distance in meters for a location match. Defaults to 20.0 m.

        Returns:
            CaracalMultipleAudioData: A container with all matching CaracalAudioData instances.
        """
        # Convert datetime objects to UTC timestamps if necessary
        if isinstance(start_time, datetime.datetime):
            start_time = self.__convert_datetime_to_utc_timestamp(start_time)
        if isinstance(end_time, datetime.datetime):
            end_time = self.__convert_datetime_to_utc_timestamp(end_time)

        all_matching_audio_data: list[CaracalAudioData] = []
        matched_session_paths: set[str] = set() # To avoid duplicate processing of sessions

        # Step 1: Scan through syslogs to match sessions based on device's median GPS location
        for syslog_container in self.sys:
            for session in syslog_container.sessions:
                if session.path in matched_session_paths:
                    continue # Already processed this session via override

                median_lon = session.header.stats.median_GPS_lon
                median_lat = session.header.stats.median_GPS_lat

                delta_lon_m = (median_lon - lon) * self.DEGREES_TO_METRES
                delta_lat_m = (median_lat - lat) * self.DEGREES_TO_METRES
                total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                if total_range < location_threshold:
                    # Session's GPS location is within threshold, now check time overlap
                    session_start_time = session.audioFiles[0].utcTime if session.audioFiles else float('inf')
                    session_end_time = session.audioFiles[-1].utcTime + session.audioFiles[-1].duration if session.audioFiles else float('-inf')

                    if max(start_time, session_start_time) < min(end_time, session_end_time):
                        multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                        for audio_data_item in multi_audio_data.stations:
                            audio_data_item.locationMatch = 'FromDevice'
                            all_matching_audio_data.append(audio_data_item)
                        matched_session_paths.add(session.path) # Mark as processed

        # Step 2: If override is present, use it to find additional matches or enforce specific ones
        if self.override is not None and self.loc is not None:
            for syslog_container in self.sys:
                for session in syslog_container.sessions:
                    if session.path in matched_session_paths:
                        continue # Already processed this session

                    session_name_from_override = self.override.getNameFromPath(session.path)
                    if session_name_from_override:
                        surveyed_pos = self.loc.fromName(session_name_from_override)
                        if surveyed_pos:
                            delta_lon_m = (surveyed_pos.lon - lon) * self.DEGREES_TO_METRES
                            delta_lat_m = (surveyed_pos.lat - lat) * self.DEGREES_TO_METRES
                            total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                            if total_range < location_threshold:
                                # Session's override location is within threshold, check time overlap
                                session_start_time = session.audioFiles[0].utcTime if session.audioFiles else float('inf')
                                session_end_time = session.audioFiles[-1].utcTime + session.audioFiles[-1].duration if session.audioFiles else float('-inf')

                                if max(start_time, session_start_time) < min(end_time, session_end_time):
                                    multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                                    for audio_data_item in multi_audio_data.stations:
                                        audio_data_item.locationMatch = 'FromOverride'
                                        audio_data_item.stationName = session_name_from_override # Set station name from override
                                        all_matching_audio_data.append(audio_data_item)
                                    matched_session_paths.add(session.path) # Mark as processed

        return CaracalMultipleAudioData(stations=all_matching_audio_data)


    def load_from_name(self, name: str,
                       start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                       location_threshold: float = 20.0) -> CaracalMultipleAudioData:
        """
        Loads audio data for a specified named location within a given time range.
        This method uses the `NamedLocationLoader` to find the coordinates for the name,
        then calls `load_from_latlon`.

        Args:
            name (str): The semantic name of the location (e.g., 'M04').
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.
            location_threshold (float): Maximum distance in meters for a location match. Defaults to 20.0 m.

        Returns:
            CaracalMultipleAudioData: A container with all matching CaracalAudioData instances.
        """
        if self.loc is None:
            print("ERR: No location info file loaded. Cannot query by name.")
            return CaracalMultipleAudioData(stations=[])

        named_location = self.loc.fromName(name)
        if named_location is None:
            print(f"ERR: No matching named location found for '{name}'.")
            return CaracalMultipleAudioData(stations=[])

        # Use the existing load_from_latlon method, which now returns CaracalMultipleAudioData
        return self.load_from_latlon(
            named_location.lat,
            named_location.lon,
            start_time,
            end_time,
            location_threshold
        )

    def load_around_latlon(self, lat: float, lon: float,
                           start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                           radius: float = 1000.0) -> CaracalMultipleAudioData:
        """
        Loads audio data from *all* stations within a specified geographical radius
        and time range. This method does not use `location_threshold` for filtering
        individual sessions but rather `radius` for a broader spatial query.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.
            radius (float): The query radius in meters. Defaults to 1000.0 m.

        Returns:
            CaracalMultipleAudioData: A container with all matching CaracalAudioData instances.
        """
        # Convert datetime objects to UTC timestamps if necessary
        if isinstance(start_time, datetime.datetime):
            start_time = self.__convert_datetime_to_utc_timestamp(start_time)
        if isinstance(end_time, datetime.datetime):
            end_time = self.__convert_datetime_to_utc_timestamp(end_time)

        all_matching_audio_data: list[CaracalAudioData] = []
        processed_session_paths: set[str] = set() # To avoid duplicate processing

        for syslog_container in self.sys:
            for session in syslog_container.sessions:
                if session.path in processed_session_paths:
                    continue # Already processed this session

                median_lon = session.header.stats.median_GPS_lon
                median_lat = session.header.stats.median_GPS_lat

                delta_lon_m = (median_lon - lon) * self.DEGREES_TO_METRES
                delta_lat_m = (median_lat - lat) * self.DEGREES_TO_METRES
                total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                if total_range < radius:
                    # Session's GPS location is within the radius, now check time overlap
                    session_start_time = session.audioFiles[0].utcTime if session.audioFiles else float('inf')
                    session_end_time = session.audioFiles[-1].utcTime + session.audioFiles[-1].duration if session.audioFiles else float('-inf')

                    if max(start_time, session_start_time) < min(end_time, session_end_time):
                        multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                        for audio_data_item in multi_audio_data.stations:
                            audio_data_item.locationMatch = 'FromDevice' # Default for load_around
                            all_matching_audio_data.append(audio_data_item)
                        processed_session_paths.add(session.path) # Mark as processed

        # Apply override logic as well, for sessions that might not have GPS or are better located by override
        if self.override is not None and self.loc is not None:
            for syslog_container in self.sys:
                for session in syslog_container.sessions:
                    if session.path in processed_session_paths:
                        continue # Already processed this session

                    session_name_from_override = self.override.getNameFromPath(session.path)
                    if session_name_from_override:
                        surveyed_pos = self.loc.fromName(session_name_from_override)
                        if surveyed_pos:
                            delta_lon_m = (surveyed_pos.lon - lon) * self.DEGREES_TO_METRES
                            delta_lat_m = (surveyed_pos.lat - lat) * self.DEGREES_TO_METRES
                            total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                            if total_range < radius:
                                # Session's override location is within radius, check time overlap
                                session_start_time = session.audioFiles[0].utcTime if session.audioFiles else float('inf')
                                session_end_time = session.audioFiles[-1].utcTime + session.audioFiles[-1].duration if session.audioFiles else float('-inf')

                                if max(start_time, session_start_time) < min(end_time, session_end_time):
                                    multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                                    for audio_data_item in multi_audio_data.stations:
                                        audio_data_item.locationMatch = 'FromOverride'
                                        audio_data_item.stationName = session_name_from_override # Set station name from override
                                        all_matching_audio_data.append(audio_data_item)
                                    processed_session_paths.add(session.path) # Mark as processed

        return CaracalMultipleAudioData(stations=all_matching_audio_data)


    def load_around_name(self, name: str,
                         start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                         radius: float = 1000.0) -> CaracalMultipleAudioData:
        """
        Loads audio data from all stations around a specified named location
        within a given time range and radius. This method uses the `NamedLocationLoader`
        to find the coordinates for the name, then calls `load_around_latlon`.

        Args:
            name (str): The semantic name of the location (e.g., 'M04').
            start_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                    for the start of the desired audio.
            end_time (float | datetime.datetime): UTC Unix timestamp (seconds) or datetime object
                                                  for the end of the desired audio.
            radius (float): The query radius in meters. Defaults to 1000.0 m.

        Returns:
            CaracalMultipleAudioData: A container with all matching CaracalAudioData instances.
        """
        if self.loc is None:
            print("ERR: No location info file loaded. Cannot query by name.")
            return CaracalMultipleAudioData(stations=[])
        if not self.loc.getAllNamedPos():
            print("ERR: No named locations loaded. Cannot query by name.")
            return CaracalMultipleAudioData(stations=[])

        named_location = self.loc.fromName(name)
        if named_location is None:
            print(f"ERR: No matching named location found for '{name}'.")
            return CaracalMultipleAudioData(stations=[])

        # Use the existing load_around_latlon method, which now returns CaracalMultipleAudioData
        return self.load_around_latlon(
            named_location.lat,
            named_location.lon,
            start_time,
            end_time,
            radius
        )

