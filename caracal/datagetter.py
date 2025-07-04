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
from scipy.io import wavfile # Need to keep this import for load_wav_legacy

# Import CaracalInventory from inventorybuilder to correctly unpickle it
from .inventorybuilder import CaracalInventory 
from .syslogparser import Identity, Stats, AudioFile, Header, Session, SyslogContainer
from .position import NamedLocation, NamedLocationLoader, OverrideLoader


# Container for multiple audio data returns
@dataclass
class CaracalQuery:
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
        multi-channel WAV file. Each station's audio data forms one or more
        channels in the output file. If a station has quadraphonic data, it
        will contribute four channels. All audio data must have the same sample rate.

        Args:
            filename (str): The filename (with or without extension) for saving.
        """
        if not self.stations:
            print("No audio data to save.")
            return

        # Determine the common sample rate and check for consistency
        sample_rate = None
        for caracal_audio in self.stations:
            if caracal_audio.audioData is not None and caracal_audio.sampleRate > 0:
                if sample_rate is None:
                    sample_rate = caracal_audio.sampleRate
                elif sample_rate != caracal_audio.sampleRate:
                    print(f"Error: Mismatched sample rates detected. Cannot save to a single multi-channel WAV.")
                    print(f"Station {caracal_audio.stationName} has sample rate {caracal_audio.sampleRate}, expected {sample_rate}.")
                    return
            else:
                print(f"Warning: Station {caracal_audio.stationName} has no valid audio data or sample rate, skipping.")

        if sample_rate is None:
            print("No valid audio data found across all stations to save.")
            return

        # Prepare data for concatenation and determine max length
        all_channels_data = []
        max_frames = 0

        for caracal_audio in self.stations:
            if caracal_audio.audioData is not None and caracal_audio.sampleRate == sample_rate:
                current_audio = caracal_audio.audioData

                # Ensure data is 2D (samples, channels) for consistent concatenation
                if current_audio.ndim == 1:
                    # Mono: reshape to (samples, 1)
                    current_audio = current_audio[:, np.newaxis]
                elif current_audio.ndim == 2:
                    # Quad or multi-channel: already (samples, channels)
                    pass
                else:
                    print(f"Warning: Skipping station {caracal_audio.stationName} due to unsupported audio data dimensions ({current_audio.ndim}).")
                    continue
                
                all_channels_data.append(current_audio)
                max_frames = max(max_frames, current_audio.shape[0])
            else:
                if caracal_audio.audioData is None:
                    print(f"Warning: Station {caracal_audio.stationName} has no audio data to save.")
                elif caracal_audio.sampleRate != sample_rate:
                     # This case is already handled by the initial check, but good to have a fallback message
                    print(f"Warning: Skipping station {caracal_audio.stationName} due to mismatched sample rate.")

        if not all_channels_data:
            print("No valid audio data to combine for multi-channel WAV.")
            return

        # Pad shorter audio segments with zeros to match the longest segment
        padded_channels = []
        for channel_data in all_channels_data:
            num_frames_to_pad = max_frames - channel_data.shape[0]
            if num_frames_to_pad > 0:
                # Pad with zeros at the end
                padding = np.zeros((num_frames_to_pad, channel_data.shape[1]), dtype=channel_data.dtype)
                padded_channels.append(np.vstack((channel_data, padding)))
            else:
                padded_channels.append(channel_data)
        
        # Concatenate all channels horizontally
        try:
            combined_audio_data = np.hstack(padded_channels)
            
            # Ensure the filename has a .wav extension
            if not filename.lower().endswith('.wav'):
                filename = f"{filename}.wav"

            sf.write(filename, combined_audio_data, sample_rate)
            print(f"Saved multi-channel audio for {len(self.stations)} stations to {filename}")
        except Exception as e:
            print(f"Error saving multi-channel audio to {filename}: {e}")

    def to_obspy_stream(self) -> obspy.Stream:
        """
        Converts the CaracalQuery instance into an ObsPy Stream object.
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

    def merge(self):
        """
        Merges CaracalAudioData instances within this CaracalQuery if they belong
        to the same station (identified by identical header.headerID) and have
        compatible sample rates. Assumes contiguous data for concatenation.

        The original 'stations' list will be replaced with the merged stations.
        """
        if not self.stations:
            print("No stations to merge.")
            return

        # Group stations by their headerID.
        # Create a hashable representation of Identity for use as dictionary key.
        grouped_stations: dict[tuple, list[CaracalAudioData]] = {}
        for station_data in self.stations:
            if station_data.header and station_data.header.headerID:
                # Create a hashable tuple from the Identity object's attributes
                header_id_tuple = (
                    station_data.header.headerID.cardID,
                    station_data.header.headerID.deviceID,
                    station_data.header.headerID.versionHash
                )
                if header_id_tuple not in grouped_stations:
                    grouped_stations[header_id_tuple] = []
                grouped_stations[header_id_tuple].append(station_data)
            else:
                print(f"WARNING: Skipping station {station_data.path} due to missing or invalid headerID for merging.")

        new_stations: list[CaracalAudioData] = []

        for header_id_tuple, group in grouped_stations.items():
            if len(group) == 1:
                # No merging needed for single-item groups
                new_stations.append(group[0])
                continue

            # Sort the group by UTCstart to ensure correct concatenation order
            group.sort(key=lambda x: x.UTCstart)

            # Initialize with the first audio data in the sorted group
            merged_audio_data = group[0]
            all_audio_arrays = [merged_audio_data.audioData]
            
            min_utc_start = merged_audio_data.UTCstart
            max_utc_end = merged_audio_data.UTCEnd
            sample_rate = merged_audio_data.sampleRate # Assume first's sample rate

            # Iterate through the rest of the group to concatenate
            for i in range(1, len(group)):
                current_audio = group[i]

                # Basic checks for compatibility before concatenating
                if current_audio.audioData is None:
                    print(f"WARNING: Skipping audio data from {current_audio.path} for merging due to None audioData.")
                    continue
                if current_audio.sampleRate != sample_rate:
                    print(f"WARNING: Skipping audio data from {current_audio.path} for merging due to differing sample rates ({current_audio.sampleRate} vs {sample_rate}).")
                    continue
                
                # Assume contiguity as per user's request, but a check for gaps could be added here
                # if current_audio.UTCstart != max_utc_end:
                #     print(f"WARNING: Gap detected between {merged_audio_data.path} (ends {max_utc_end}) and {current_audio.path} (starts {current_audio.UTCstart}).")

                all_audio_arrays.append(current_audio.audioData)
                max_utc_end = max(max_utc_end, current_audio.UTCEnd) # Update overall end time

            # Perform the actual concatenation
            if all_audio_arrays and all(arr is not None for arr in all_audio_arrays):
                try:
                    # Concatenate along the appropriate axis (0 for mono, 0 for multi-channel if (samples, channels))
                    # Assuming audioData is 1D for mono or 2D for quad (samples, channels)
                    if all_audio_arrays[0].ndim == 1: # Mono
                        merged_audio_data.audioData = np.concatenate(all_audio_arrays, axis=0)
                    elif all_audio_arrays[0].ndim == 2: # Quad
                        merged_audio_data.audioData = np.concatenate(all_audio_arrays, axis=0)
                    else:
                        print(f"WARNING: Cannot concatenate audio data for headerID {header_id_tuple} due to unsupported dimensions: {all_audio_arrays[0].ndim}")
                        merged_audio_data.audioData = None # Invalidate audioData if concatenation fails
                        continue # Skip adding this merged data if concatenation failed

                    # Update metadata for the merged object
                    merged_audio_data.UTCstart = min_utc_start
                    merged_audio_data.UTCEnd = max_utc_end
                    
                    # Update audioFile details to reflect the merged segment
                    # Create a new AudioFile instance or modify existing one to reflect merged properties
                    merged_audio_data.audioFile = AudioFile(
                        utcTime=float(min_utc_start),
                        duration=float(max_utc_end - min_utc_start),
                        subpath="merged_data_segment" # Indicate it's a merged segment
                    )
                    merged_audio_data.sampleRate = sample_rate # Ensure sample rate is consistent
                    merged_audio_data.locationMatch = group[0].locationMatch # Indicate it's merged data
                    
                    new_stations.append(merged_audio_data)

                except Exception as e:
                    print(f"ERROR: Failed to concatenate audio for headerID {header_id_tuple}: {e}")
            else:
                print(f"WARNING: No valid audio arrays to concatenate for headerID {header_id_tuple}.")

        self.stations = new_stations
        print(f"Merged stations. Original count: {len(grouped_stations)} unique stations. New count: {len(self.stations)}.")



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
                 syslogdata: str | None = None, # Made optional
                 locationinfo: str | None = None, # Made optional
                 timezone: str = "UTC",
                 audio_mode: str = 'mono',
                 overrideinfo: str | None = None,
                 merge: bool = False):
        """
        Initializes a DataGetter instance.

        Args:
            rootpath (str): The main, top-level path where all the data lives.
                            Typically, this will be the same path as used for InventoryBuilder.
            syslogdata (str, optional): Path to a pickled file containing a list of SyslogContainer objects.
                                        Defaults to 'inventory.pkl' inside `rootpath`.
            locationinfo (str, optional): Path to a CSV file for semantic/surveyed locations.
                                          Used by NamedLocationLoader. Defaults to 'locations.csv' inside `rootpath`.
            timezone (str): Timezone string (ZoneInfo code, e.g., "America/New_York").
                            Used to convert naive datetimes to aware timestamps. Defaults to "UTC".
            audio_mode (str): Specifies the audio channel mode. 'mono' for single channel,
                              quad' for four channels. Defaults to 'mono'.
            overrideinfo (str, optional): Path to a CSV file for overriding semantic/surveyed locations.
                                          Used by OverrideLoader. Defaults to None.
            merge (bool): If True, will attempt to merge contiguous audio segments from the same station e.g. if they span an hourly boundary. This does remove detailed timing information.

        Raises:
            ValueError: If an invalid `audio_mode` is provided.
        """
        self.rootpath = rootpath
        self.loc: NamedLocationLoader | None = None
        self.override: OverrideLoader | None = None
        self.merge = merge

        if audio_mode not in ['mono', 'quad']:
            raise ValueError("Audio mode must be 'mono' or 'quad'")
        self.audio_mode = audio_mode
        print(f"DataGetter: Audio Mode set to {self.audio_mode}")

        self.timezone_str = timezone
        self.local_timezone = ZoneInfo(timezone)
        print(f"DataGetter: Timezone set to: {self.local_timezone}")

        # Determine syslogdata path
        syslogdata_path = syslogdata if syslogdata is not None else os.path.join(self.rootpath, "inventory.pkl")
        
        # Load the parsed syslogs from the pickled file
        try:
            with open(syslogdata_path, 'rb') as handle:
                loaded_inventory = pickle.load(handle)
                # Check if the loaded object is a CaracalInventory and extract syslog_containers
                if isinstance(loaded_inventory, CaracalInventory):
                    self.sys: list[SyslogContainer] = loaded_inventory.syslog_containers
                    print(f"DataGetter: Loaded {len(self.sys)} syslog entries from {syslogdata_path}.")
                else:
                    print(f"ERROR: Expected CaracalInventory object from {syslogdata_path}, but got {type(loaded_inventory)}. Trying to load as raw list.")
                    # Fallback for older inventory files that might be just a list
                    self.sys: list[SyslogContainer] = loaded_inventory
                    print(f"DataGetter: Loaded {len(self.sys)} syslog entries (as raw list) from {syslogdata_path}.")

        except FileNotFoundError:
            print(f"ERROR: Syslog data file not found at {syslogdata_path}. Please ensure it exists.")
            self.sys = []
        except Exception as e:
            print(f"ERROR: Failed to load syslog data from {syslogdata_path}: {e}")
            self.sys = []

        # Determine locationinfo path
        locationinfo_path = locationinfo if locationinfo is not None else os.path.join(self.rootpath, "locations.csv")

        if os.path.exists(locationinfo_path):
            try:
                self.loc = NamedLocationLoader(locationinfo_path)
            except Exception as e:
                print(f"ERROR: Could not initialize NamedLocationLoader with {locationinfo_path}: {e}")
        else:
            print(f"INFO: Location info file not found at {locationinfo_path}. Named location queries may not work.")

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
    def __unpack(datx):
        '''Internal function to inflate a packed wave file into four channels'''
        num_samples = np.shape(datx)[0]
        new_dat = np.empty((num_samples,4),dtype=np.int32)
        gain_array = np.empty(num_samples)
        dat_int = datx.astype(np.int32,copy=True)
        MASK_A = np.uint32(0xFFFFFE00)
        MASK_B = np.uint32(0xFFF00000)
        MASK_C = np.uint32(0x000FFF00)
        MASK_D1 = np.uint32(0x000000FF)
        MASK_D0 = np.uint32(0x0F)
        MASK_GAIN = np.uint32(0x1f0)
        for idx in np.arange(num_samples):
            d = dat_int[idx]
            ch_a = d[0]& MASK_A
            ch_b = (d[1]& MASK_B)>>20
            ch_c = (d[1]& MASK_C)>>8
            ch_d = (d[1]&MASK_D1)<<4 | (d[0]&MASK_D0)
            gain = (d[0]&MASK_GAIN)>>4
            # unsigned to signed conversion for the mantissa
            ch_a = (ch_a ^ 0x80000000) - 0x80000000
            ch_b = (ch_b ^ 0x800) - 0x800
            ch_c = (ch_c ^ 0x800) - 0x800
            ch_d = (ch_d ^ 0x800) - 0x800
            # and now the exponent
            ch_b = ch_b * 2**gain
            ch_c = ch_c * 2**gain
            ch_d = ch_d * 2**gain
            # ACM 16APR24 - update the unpacker to get the angles in order:
            #  mic_angle = [0,90,180,270]
            new_dat[idx,:] = [ch_a,ch_b,ch_d,ch_c]
            gain_array[idx] = gain
        return new_dat,gain_array

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
        print("load_wav",audio_mode)
        if duration is None:
            raise ValueError("Need a duration to load audio.")

        try:
            with sf.SoundFile(filename, 'r') as track:
                if not track.seekable():
                    raise ValueError(f"Audio file '{filename}' is not seekable.")
                print(f"1.Loading WAV file '{filename}' with audio mode '{audio_mode}'.")
                sr = track.samplerate
                start_frame = int((start_offset or 0) * sr)
                frames_to_read = int(sr * duration)
                print(start_frame)
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
                    print("quad mode selected, reading 4 channels.")
                    # explicit int32 for bit manipulation
                    audio_section = track.read(frames_to_read,dtype='int32')
                    print(np.shape(audio_section))
                    quad_audio, gain = DataGetter.__unpack(audio_section)
                    print(np.shape(quad_audio))
                    # scaling back to float64 [-1,1] range
                    audio_float = np.array(quad_audio).astype(np.float64)
                    audio_scaled = quad_audio/2**31
                    return sr,audio_scaled
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
            # wavfile.read can return data as int16 or float, depending on the WAV file format
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

    def get_audio_from_session(self, session: Session, start_timestamp: float, end_timestamp: float) -> CaracalQuery:
        """
        Retrieves audio data for a specified time range from a given session.
        This method can span multiple hourly-aligned audio files within a session
        if the query time range crosses file boundaries.

        Args:
            session (Session): The session object from which to extract audio.
            start_timestamp (float): UTC Unix timestamp (seconds) for the start of the desired audio.
            end_timestamp (float): UTC Unix timestamp (seconds) for the end of the desired audio.

        Returns:
            CaracalQuery: A container with the requested audio data.
                                      Will contain one or more CaracalAudioData instances,
                                      concatenated if they are contiguous and from the same station.
        """
        matching_audio_data_list: list[CaracalAudioData] = []
        
        # Filter audio files within the session that overlap with the query time range
        relevant_audio_files = []
        for audiofile in session.audioFiles:
            try:
                # Explicitly cast to float to prevent TypeError if data is corrupted
                file_utc_start = float(audiofile.utcTime)
                file_utc_end = float(audiofile.utcTime) + float(audiofile.duration)
            except (ValueError, TypeError) as e:
                print(f"WARNING: Skipping audio file due to corrupted time data: {audiofile.subpath} - {e}")
                continue # Skip this audiofile if its time data is not convertible to float

            # Check for overlap: [start_timestamp, end_timestamp] overlaps with [file_utc_start, file_utc_end]
            if max(start_timestamp, file_utc_start) < min(end_timestamp, file_utc_end):
                relevant_audio_files.append(audiofile)
        
        # Sort relevant files by their UTC start time to ensure correct concatenation order
        relevant_audio_files.sort(key=lambda af: af.utcTime)

        # Process each relevant audio file
        for audiofile in relevant_audio_files:
            try:
                # Re-cast here for safety, though it should be float after filtering
                file_utc_start = float(audiofile.utcTime)
                file_utc_end = float(audiofile.utcTime) + float(audiofile.duration)
            except (ValueError, TypeError) as e:
                print(f"WARNING: Skipping audio file during processing due to corrupted time data: {audiofile.subpath} - {e}")
                continue

            # Determine the actual segment to load from this specific file
            segment_start_utc = max(start_timestamp, file_utc_start)
            segment_end_utc = min(end_timestamp, file_utc_end)

            if segment_start_utc >= segment_end_utc:
                # No actual overlap for this file, or segment is zero length
                continue

            offset_in_file = segment_start_utc - file_utc_start
            duration_to_load = segment_end_utc - segment_start_utc

            full_filename = os.path.join(self.rootpath, session.path, audiofile.subpath)
            print(full_filename, offset_in_file, duration_to_load, self.audio_mode)
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

        return CaracalQuery(stations=matching_audio_data_list)


    def load_from_session(self, session: Session, start_time: float | datetime.datetime, end_time: float | datetime.datetime) -> CaracalQuery:
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
            CaracalQuery: A container with the requested audio data.
                                      Will contain one or more CaracalAudioData instances.
        """
        if not isinstance(session, Session):
            print("ERR: Input is not a Session instance.")
            return CaracalQuery(stations=[]) # Return empty container

        # Convert datetime objects to UTC timestamps if necessary
        if isinstance(start_time, datetime.datetime):
            start_time = self.__convert_datetime_to_utc_timestamp(start_time)
        if isinstance(end_time, datetime.datetime):
            end_time = self.__convert_datetime_to_utc_timestamp(end_time)

        # Use the updated get_audio_from_session which handles multi-file queries
        return self.get_audio_from_session(session, start_time, end_time)


    def load_from_device_id(self, devID: str, start_time: float | datetime.datetime, end_time: float | datetime.datetime) -> CaracalQuery:
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
            CaracalQuery: A container with all matching CaracalAudioData instances.
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
                session_start_time = float('inf')
                session_end_time = float('-inf')

                if session.audioFiles:
                    try:
                        # Explicitly cast to float for safety
                        first_file_utc = float(session.audioFiles[0].utcTime)
                        last_file_utc = float(session.audioFiles[-1].utcTime)
                        last_file_duration = float(session.audioFiles[-1].duration)
                        session_start_time = first_file_utc
                        session_end_time = last_file_utc + last_file_duration
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Skipping session {session.path} due to corrupted audio file time data: {e}")
                        # Keep session_start_time as inf and session_end_time as -inf
                        # so this session won't match any time queries.
                        continue # Skip to the next session if time data is bad

                session_dev_id = session.header.headerID.deviceID
                if devID == session_dev_id:
                    # Found a session matching the device ID, now check time overlap
                    if max(start_time, session_start_time) < min(end_time, session_end_time):
                        # The session itself overlaps with the query time
                        # Call get_audio_from_session to extract relevant audio segments
                        # This will handle spanning multiple files within that session
                        multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                        all_matching_audio_data.extend(multi_audio_data.stations)
        
        return CaracalQuery(stations=all_matching_audio_data)
    

    def load_around_latlon(self, lat: float, lon: float,
                           start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                           radius: float = 1000.0) -> CaracalQuery:
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
            CaracalQuery: A container with all matching CaracalAudioData instances.
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

                # Initialize session times to values that won't match if data is bad
                session_start_time = float('inf')
                session_end_time = float('-inf')

                if session.audioFiles:
                    try:
                        # Explicitly cast to float for safety
                        first_file_utc = float(session.audioFiles[0].utcTime)
                        last_file_utc = float(session.audioFiles[-1].utcTime)
                        last_file_duration = float(session.audioFiles[-1].duration)
                        session_start_time = first_file_utc
                        session_end_time = last_file_utc + last_file_duration
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Skipping session {session.path} due to corrupted audio file time data: {e}")
                        continue # Skip to the next session if time data is bad

                delta_lon_m = (median_lon - lon) * self.DEGREES_TO_METRES
                delta_lat_m = (median_lat - lat) * self.DEGREES_TO_METRES
                total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                if total_range < radius:
                    # Session's GPS location is within the radius, now check time overlap
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
                                # Initialize session times to values that won't match if data is bad
                                session_start_time = float('inf')
                                session_end_time = float('-inf')

                                if session.audioFiles:
                                    try:
                                        # Explicitly cast to float for safety
                                        first_file_utc = float(session.audioFiles[0].utcTime)
                                        last_file_utc = float(session.audioFiles[-1].utcTime)
                                        last_file_duration = float(session.audioFiles[-1].duration)
                                        session_start_time = first_file_utc
                                        session_end_time = last_file_utc + last_file_duration
                                    except (ValueError, TypeError) as e:
                                        print(f"WARNING: Skipping session {session.path} due to corrupted audio file time data: {e}")
                                        continue # Skip to the next session if time data is bad

                                # Session's override location is within radius, check time overlap
                                if max(start_time, session_start_time) < min(end_time, session_end_time):
                                    multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                                    for audio_data_item in multi_audio_data.stations:
                                        audio_data_item.locationMatch = 'FromOverride'
                                        audio_data_item.stationName = session_name_from_override # Set station name from override
                                        all_matching_audio_data.append(audio_data_item)
                                    processed_session_paths.add(session.path) # Mark as processed

        return CaracalQuery(stations=all_matching_audio_data)


    def load_from_latlon(self, lat: float, lon: float,
                         start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                         location_threshold: float = 20.0) -> CaracalQuery:
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
            CaracalQuery: A container with all matching CaracalAudioData instances.
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

                # Initialize session times to values that won't match if data is bad
                session_start_time = float('inf')
                session_end_time = float('-inf')

                if session.audioFiles:
                    try:
                        # Explicitly cast to float for safety
                        first_file_utc = float(session.audioFiles[0].utcTime)
                        last_file_utc = float(session.audioFiles[-1].utcTime)
                        last_file_duration = float(session.audioFiles[-1].duration)
                        session_start_time = first_file_utc
                        session_end_time = last_file_utc + last_file_duration
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Skipping session {session.path} due to corrupted audio file time data: {e}")
                        continue # Skip to the next session if time data is bad

                delta_lon_m = (median_lon - lon) * self.DEGREES_TO_METRES
                delta_lat_m = (median_lat - lat) * self.DEGREES_TO_METRES
                total_range = np.sqrt(delta_lon_m**2 + delta_lat_m**2)

                if total_range < location_threshold:
                    # Session's GPS location is within threshold, now check time overlap
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
                                # Initialize session times to values that won't match if data is bad
                                session_start_time = float('inf')
                                session_end_time = float('-inf')

                                if session.audioFiles:
                                    try:
                                        # Explicitly cast to float for safety
                                        first_file_utc = float(session.audioFiles[0].utcTime)
                                        last_file_utc = float(session.audioFiles[-1].utcTime)
                                        last_file_duration = float(session.audioFiles[-1].duration)
                                        session_start_time = first_file_utc
                                        session_end_time = last_file_utc + last_file_duration
                                    except (ValueError, TypeError) as e:
                                        print(f"WARNING: Skipping session {session.path} due to corrupted audio file time data: {e}")
                                        continue # Skip to the next session if time data is bad

                                # Session's override location is within radius, check time overlap
                                if max(start_time, session_start_time) < min(end_time, session_end_time):
                                    multi_audio_data = self.get_audio_from_session(session, start_time, end_time)
                                    for audio_data_item in multi_audio_data.stations:
                                        audio_data_item.locationMatch = 'FromOverride'
                                        audio_data_item.stationName = session_name_from_override # Set station name from override
                                        all_matching_audio_data.append(audio_data_item)
                                    matched_session_paths.add(session.path) # Mark as processed
        if self.merge:
            return  CaracalQuery(stations=all_matching_audio_data).merge()
        return CaracalQuery(stations=all_matching_audio_data)


    def load_around_name(self, name: str,
                         start_time: float | datetime.datetime, end_time: float | datetime.datetime,
                         radius: float = 1000.0) -> CaracalQuery:
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
            CaracalQuery: A container with all matching CaracalAudioData instances.
        """
        if self.loc is None:
            print("ERR: No location info file loaded. Cannot query by name.")
            return CaracalQuery(stations=[])
        if not self.loc.getAllNamedPos():
            print("ERR: No named locations loaded. Cannot query by name.")
            return CaracalQuery(stations=[])

        named_location = self.loc.fromName(name)
        if named_location is None:
            print(f"ERR: No matching named location found for '{name}'.")
            return CaracalQuery(stations=[])

        # Use the existing load_around_latlon method, which now returns CaracalQuery
        return self.load_around_latlon(
            named_location.lat,
            named_location.lon,
            start_time,
            end_time,
            radius
        )