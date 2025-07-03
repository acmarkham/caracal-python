import numpy as np
import os
from dataclasses import dataclass, field, InitVar

@dataclass
class Identity:
    """
    Represents the unique identity information of a device and SD card.

    Attributes:
        cardID (str): The unique identifier of the SD card.
        deviceID (str): The unique identifier of the device.
        versionHash (str): The firmware version hash of the device.
    """
    cardID: str = ''
    deviceID: str = ''
    versionHash: str = ''

@dataclass
class PPS:
    """
    Stores Pulse Per Second (PPS) timing information, associating a UTC timestamp
    with a system timestamp.

    Attributes:
        UTCTime (float): The UTC time of the PPS pulse.
        sysTime (float): The system time corresponding to the PPS pulse.
    """
    UTCTime: float = 0.0
    sysTime: float = 0.0

@dataclass
class Stats:
    """
    Contains summary statistics for a given session, including message counts,
    file counts, GPS fix details, timing information, and voltage readings.

    Attributes:
        num_msg (int): Total number of messages processed in the session.
        num_files (int): Total number of audio files recorded in the session.
        num_fixes (int): Number of valid GPS fixes obtained.
        num_timing (int): Number of PPS timing synchronization points.
        ppsFirst (PPS): PPS information for the first synchronization point.
        ppsMid (PPS): PPS information for a mid-point synchronization.
        ppsLast (PPS): PPS information for the last synchronization point.
        median_GPS_lon (float): Median GPS longitude for the session.
        median_GPS_lat (float): Median GPS latitude for the session.
        start_voltage (float): Average voltage at the beginning of the session.
        end_voltage (float): Average voltage at the end of the session.
    """
    num_msg: int = 0
    num_files: int = 0
    num_fixes: int = 0
    num_timing: int = 0
    ppsFirst: PPS = field(default_factory=PPS)
    ppsMid: PPS = field(default_factory=PPS)
    ppsLast: PPS = field(default_factory=PPS)
    median_GPS_lon: float = 0.0
    median_GPS_lat: float = 0.0
    start_voltage: float = 0.0
    end_voltage: float = 0.0

@dataclass
class Header:
    """
    Encapsulates common header and metadata information for a recording session.

    Attributes:
        headerID (Identity): Device and SD card identity information.
        stats (Stats): Summary statistics for the session.
        PPSpoints (list[PPS]): A list of all PPS synchronization points recorded.
        sysDuration (float): The total duration of the session in system time.
    """
    headerID: Identity = field(default_factory=Identity)
    stats: Stats = field(default_factory=Stats)
    PPSpoints: list[PPS] = field(default_factory=list)
    sysDuration: float = 0.0

@dataclass
class GPSlocation:
    """
    Stores individual GPS location data.

    Attributes:
        lon (float): Longitude.
        lat (float): Latitude.
        gpsTime (int): GPS time.
        sysTime (int): System time when the GPS fix was recorded.
    """
    lon: float = 0.0
    lat: float = 0.0
    gpsTime: int = 0
    sysTime: int = 0

@dataclass
class AudioTiming:
    """
    Stores highly detailed timing information for a single audio file, enabling
    sample-accurate alignment to GPS PPS and identification of timing degradation.

    Attributes:
        ppsBefore (PPS): PPS information just before the audio file's start.
        ppsAfter (PPS): PPS information just after the audio file's end.
        ppsDuring (PPS): PPS information at a midpoint during the audio file.
        estimatedEndSystime (float): Estimated system timestamp for the end of the file.
        blockStartSysTime (float): Precise ISR (Interrupt Service Routine) system
                                   timestamp for the start of the audio block.
        blockEndSysTime (float): Precise ISR system timestamp for the end of the audio block.
        blockLen (int): Number of samples (frames) per audio block.
        numBlocks (int): Total number of blocks written to the file.
        numSamples (int): Total number of samples (can be multichannel) written to the file.
    """
    ppsBefore: PPS = field(default_factory=PPS)
    ppsAfter: PPS = field(default_factory=PPS)
    ppsDuring: PPS = field(default_factory=PPS)
    estimatedEndSystime: float = 0.0
    blockStartSysTime: float = 0.0
    blockEndSysTime: float = 0.0
    blockLen: int = 0
    numBlocks: int = 0
    numSamples: int = 0

@dataclass
class AudioFile:
    """
    Stores comprehensive information about a single audio file.

    Attributes:
        subpath (str): The subfolder path where the audio file is located.
        duration (int): The nominal duration of the audio file in seconds (e.g., 3600).
        utcTime (float): The UTC timestamp of the file's creation, nominally aligned to the hour.
        creation_sysTime (float): The local system timestamp when the file was created.
        timing (AudioTiming): Detailed timing information specific to this audio file.
    """
    subpath: str = ''
    duration: int = 0
    utcTime: float = 0.0
    creation_sysTime: float = 0.0
    timing: AudioTiming = field(default_factory=AudioTiming)

@dataclass
class Session:
    """
    Represents a contiguous group of data from a syslog. New sessions are
    typically triggered by reset events (e.g., power cycles or internal resets).

    Attributes:
        path (str): The relative top-level path of the syslog.
        sessionID (int): A monotonically increasing integer ID for the session.
        header (Header): Header and metadata information for this session.
        audioFiles (list[AudioFile]): A list of all audio files recorded within this session.
    """
    path: str = ''
    sessionID: int = 0
    header: Header = field(default_factory=Header)
    audioFiles: list[AudioFile] = field(default_factory=list)

@dataclass
class SyslogContainer:
    """
    The parent container class that holds all information parsed from a single
    syslog.txt file. This can encompass data spanning multiple device resets or
    power-downs.

    Attributes:
        path (str): The relative top-level path to the syslog file's directory.
        filename (str): The name of the syslog file.
        sessions (list[Session]): A list of parsed sessions within the syslog.
        parserVersion (int): The version number of the parser used.
    """
    path: str = ''
    filename: str = ''
    sessions: list[Session] = field(default_factory=list)
    parserVersion: int = 0



class SyslogParser:
    """
    Parses syslog files to extract and organize device and session information,
    including timing, GPS, and audio file metadata.
    """

    # Constants for audio file properties
    DURATION: int = 3600  # Nominal audio file length in seconds
    ALIGNMENT: int = 3600 # Audio files should start/end on this boundary
    SAMPLES_PER_BLOCK: int = 1536 # Number of audio frames per block (HARDCODED)
    NOMINAL_SAMPLE_RATE: int = 16000 # Nominal sample rate (HARDCODED)
    PARSER_VERSION: int = 1

    def __init__(self, filename: str):
        """
        Initializes the SyslogParser with the path to the syslog file.

        Args:
            filename (str): The full path to the syslog file to be parsed.
        """
        self.filename: str = filename

    def process(self, detailed_PPS: bool = False) -> SyslogContainer | None:
        """
        Processes the syslog file specified during initialization and organizes
        its contents into a SyslogContainer object.

        Args:
            detailed_PPS (bool): If True, a detailed list of all PPS points
                                 will be included in the session header.

        Returns:
            SyslogContainer | None: An object containing all parsed session data,
                                    or None if no sessions are found.
        """
        raw_sessions = self.__parse_syslog(self.filename)
        if raw_sessions is None:
            return None

        sys_container = SyslogContainer(
            path=os.path.split(self.filename)[0],
            filename=os.path.split(self.filename)[1],
            parserVersion=self.PARSER_VERSION
        )

        for s_count, session_data in enumerate(raw_sessions):
            my_session = Session(
                sessionID=s_count,
                path=sys_container.path
            )

            # Populate session header duration
            time_stamp_start = session_data[0]['sysTime']
            time_stamp_end = session_data[-1]['sysTime']
            my_session.header.sysDuration = time_stamp_end - time_stamp_start

            # Populate identity
            my_session.header.headerID.cardID = self.extract_card_id(session_data)
            my_session.header.headerID.deviceID = self.extract_node_id(session_data)
            my_session.header.headerID.versionHash = self.extract_firmware_hash(session_data)

            # Populate stats
            my_session.header.stats.num_msg = len(session_data)

            # GPS stats
            lons, lats, gps_times = self.extract_positions(session_data)
            my_session.header.stats.median_GPS_lat = np.nanmedian(lats) if lats else 0.0
            my_session.header.stats.median_GPS_lon = np.nanmedian(lons) if lons else 0.0
            my_session.header.stats.num_fixes = len(gps_times)

            # Timing stats
            gps_marks, pps_measures, _, _ = self.extract_timing(session_data)
            my_session.header.stats.num_timing = len(gps_marks)
            if gps_marks:
                my_session.header.stats.ppsFirst.UTCTime = gps_marks[0]
                my_session.header.stats.ppsFirst.sysTime = pps_measures[0]
                my_session.header.stats.ppsLast.UTCTime = gps_marks[-1]
                my_session.header.stats.ppsLast.sysTime = pps_measures[-1]
                mid_idx = int(len(gps_marks) / 2)
                my_session.header.stats.ppsMid.UTCTime = gps_marks[mid_idx]
                my_session.header.stats.ppsMid.sysTime = pps_measures[mid_idx]

                if detailed_PPS:
                    for pps_gps, pps_sys in zip(gps_marks, pps_measures):
                        pps = PPS(UTCTime=pps_gps, sysTime=pps_sys)
                        my_session.header.PPSpoints.append(pps)

            # Voltage stats
            voltages, _ = self.extract_voltages(session_data)
            try:
                if voltages:
                    my_session.header.stats.end_voltage = np.mean(voltages[-10:-1])
                    my_session.header.stats.start_voltage = np.mean(voltages[0:10])
            except Exception as e:
                print(f"WARN: Could not extract voltage stats for session {s_count}: {e}")
                pass # Continue processing even if voltage extraction fails

            # EOF info
            eof_estimated, blk_seq, h_systime, f_systime = self.extract_eof(session_data)

            # AudioFiles
            filenames, systimes = self.extract_files(session_data)
            my_session.header.stats.num_files = len(filenames)

            for file_path, creation_systime in zip(filenames, systimes):
                filename_only = os.path.splitext(os.path.split(file_path)[1])[0]
                try:
                    # Assuming filename format: "prefix_TIMESTAMP.wav"
                    file_utc_time = int(filename_only.split('_')[1])
                except (IndexError, ValueError):
                    print(f"WARN: Error extracting file UTC time from filename: {filename_only}. Setting to 0.")
                    file_utc_time = 0

                audio_file = AudioFile(
                    subpath=file_path,
                    duration=self.DURATION,
                    utcTime=float(file_utc_time),
                    creation_sysTime=creation_systime
                )

                # Populate PPS info for audio file
                if audio_file.utcTime != 0 and len(gps_marks) > 5:
                    gps_mark_np = np.array(gps_marks)
                    pps_meas_np = np.array(pps_measures)

                    # PPS Before
                    idx_before = np.searchsorted(gps_mark_np, audio_file.utcTime, side="left")
                    if idx_before > 0:
                        audio_file.timing.ppsBefore.UTCTime = gps_mark_np[idx_before - 1]
                        audio_file.timing.ppsBefore.sysTime = pps_meas_np[idx_before - 1]

                    # PPS After
                    idx_after = np.searchsorted(gps_mark_np, audio_file.utcTime + audio_file.duration, side="right")
                    if idx_after < len(gps_mark_np):
                        audio_file.timing.ppsAfter.UTCTime = gps_mark_np[idx_after]
                        audio_file.timing.ppsAfter.sysTime = pps_meas_np[idx_after]

                    # PPS During (if enough points)
                    if (idx_after - idx_before) > 2:
                        idx_during = int((idx_after - idx_before) / 2 + idx_before)
                        audio_file.timing.ppsDuring.UTCTime = gps_mark_np[idx_during]
                        audio_file.timing.ppsDuring.sysTime = pps_meas_np[idx_during]

                # Populate EOF info for audio file
                if len(eof_estimated) > 2:
                    eof_idx_before = np.searchsorted(eof_estimated, creation_systime, side="left")
                    if eof_idx_before > 0:
                        eof_idx_before -= 1
                    else:
                        # First file in session, use initial EOF data as estimates
                        audio_file.timing.estimatedEndSystime = eof_estimated[0]
                        audio_file.timing.numBlocks = blk_seq[0]
                        audio_file.timing.blockLen = self.SAMPLES_PER_BLOCK
                        audio_file.timing.numSamples = audio_file.timing.numBlocks * self.SAMPLES_PER_BLOCK
                        eof_idx_before = None # Indicate no suitable 'before' EOF found

                    eof_idx_after = None
                    if eof_idx_before is not None and (eof_idx_before + 1) < len(eof_estimated):
                        eof_idx_after = eof_idx_before + 1

                    if eof_idx_before is not None and eof_idx_after is not None:
                        actual_start = max(h_systime[eof_idx_before], f_systime[eof_idx_before])
                        actual_end = max(h_systime[eof_idx_after], f_systime[eof_idx_after])
                        num_blocks = blk_seq[eof_idx_after] - blk_seq[eof_idx_before]
                        num_samples = num_blocks * self.SAMPLES_PER_BLOCK

                        audio_file.timing.estimatedEndSystime = eof_estimated[eof_idx_after]
                        audio_file.timing.blockStartSysTime = actual_start
                        audio_file.timing.blockEndSysTime = actual_end
                        audio_file.timing.numBlocks = num_blocks
                        audio_file.timing.blockLen = self.SAMPLES_PER_BLOCK
                        audio_file.timing.numSamples = num_samples

                my_session.audioFiles.append(audio_file)

            sys_container.sessions.append(my_session)

        return sys_container

    @staticmethod
    def __parse_line(linestring: str) -> dict | None:
        """
        Parses a single line from the syslog.

        Args:
            linestring (str): A single line from the syslog file.

        Returns:
            dict | None: A dictionary containing 'msgCount', 'sysTime', and 'rawmsg'
                         if the line is a valid syslog message (starts with 'SYS'),
                         otherwise None.
        """
        fields = linestring.strip().split(',', maxsplit=3)
        if fields[0] != 'SYS':
            return None
        try:
            msg_count = int(fields[1])
            sys_time = int(fields[2]) / 1e6
            raw_msg = fields[3]
            return {'msgCount': msg_count, 'sysTime': sys_time, 'rawmsg': raw_msg}
        except (ValueError, IndexError) as e:
            print(f"ERR: Failed to parse line: '{linestring.strip()}' - {e}")
            return None

    def __parse_syslog(self, filename: str) -> list[list[dict]] | None:
        """
        Parses the entire syslog file and separates messages into sessions
        based on 'msgCount' resetting to 1.

        Args:
            filename (str): The path to the syslog file.

        Returns:
            list[list[dict]] | None: A list of lists, where each inner list
                                     represents a session (a sequence of parsed
                                     log messages), or None if the file cannot
                                     be opened or parsed.
        """
        sessions: list[list[dict]] = []
        current_session: list[dict] = []
        
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parsed_line = self.__parse_line(line)
                    if parsed_line:
                        if parsed_line['msgCount'] == 1:
                            if current_session: # If there's an existing session, append it
                                sessions.append(current_session)
                            current_session = [parsed_line] # Start new session
                        else:
                            current_session.append(parsed_line)
            if current_session: # Append the last session
                sessions.append(current_session)
            
            return sessions if sessions else None
        except FileNotFoundError:
            print(f"ERROR: Syslog file not found: {filename}")
            return None
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while reading syslog '{filename}': {e}")
            return None

    @staticmethod
    def extract_node_id(session: list[dict]) -> str | None:
        """
        Extracts the device ID from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            str | None: The concatenated device ID string, or None if not found.
        """
        for line in session:
            if line['rawmsg'].startswith('DEVICE_ID'):
                fields = line['rawmsg'].split(" ")
                if len(fields) >= 4:
                    return "".join(fields[1:4])
        return None

    @staticmethod
    def extract_card_id(session: list[dict]) -> str | None:
        """
        Extracts the SD card ID from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            str | None: The SD card ID string, or None if not found.
        """
        for line in session:
            if line['rawmsg'].startswith('CARD_ID'):
                fields = line['rawmsg'].split(" ")
                if len(fields) >= 2:
                    return fields[1].strip()
        return None

    @staticmethod
    def extract_firmware_hash(session: list[dict]) -> str | None:
        """
        Extracts the firmware hash (version) from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            str | None: The firmware hash string, or None if not found.
        """
        for line in session:
            if line['rawmsg'].startswith('Flash CRC:'):
                fields = line['rawmsg'].split(" ")
                if len(fields) >= 3:
                    return fields[2].strip()
        return None

    @staticmethod
    def extract_positions(session: list[dict]) -> tuple[list[float], list[float], list[float]]:
        """
        Extracts all GPS location fixes (longitude, latitude, and system time)
        from a session's log messages. Only extracts fixes with type > 1.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[float], list[float], list[float]]: Three lists containing
            longitudes, latitudes, and system timestamps of GPS fixes, respectively.
        """
        lons: list[float] = []
        lats: list[float] = []
        sys_times: list[float] = []
        for line in session:
            if line['rawmsg'].startswith('GPS_NAV2'):
                try:
                    data = line['rawmsg']
                    fields = data.split(",")
                    fix_type = fields[2]
                    if int(fix_type[-1]) > 1: # Assuming fixType format is like "fixType:X"
                        lon = float(fields[5].split(':')[1]) / 1e7
                        lat = float(fields[6].split(':')[1]) / 1e7
                        lons.append(lon)
                        lats.append(lat)
                        sys_times.append(line['sysTime'])
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting GPS position from line: '{line['rawmsg'].strip()}' - {e}")
        return lons, lats, sys_times

    @staticmethod
    def extract_files(session: list[dict]) -> tuple[list[str], list[float]]:
        """
        Extracts all audio file paths and their creation system timestamps
        from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[str], list[float]]: Two lists containing audio file paths
            and their corresponding system timestamps.
        """
        sys_times: list[float] = []
        filenames: list[str] = []
        for line in session:
            if line['rawmsg'].startswith('Newfile:'):
                try:
                    data = line['rawmsg']
                    fields = data.split(':', maxsplit=1) # Split only on the first colon
                    if len(fields) > 1:
                        filename = fields[1].strip()
                        filenames.append(filename)
                        sys_times.append(line['sysTime'])
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting file info from line: '{line['rawmsg'].strip()}' - {e}")
        return filenames, sys_times

    @staticmethod
    def extract_timing(session: list[dict]) -> tuple[list[int], list[float], list[float], list[float]]:
        """
        Extracts GPS timing synchronization information (UTC seconds, actual PPS,
        estimated PPS, and drift) from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[int], list[float], list[float], list[float]]: Four lists
            containing GPS UTC marks, measured PPS system times, estimated PPS
            system times, and PPS drift in milliseconds, respectively.
        """
        gps_mark_list: list[int] = []
        pps_meas_list: list[float] = []
        pps_est_list: list[float] = []
        pps_drift_list: list[float] = []
        for line in session:
            if line['rawmsg'].startswith('GPS: UTC sec'):
                try:
                    data = line['rawmsg']
                    # Example: GPS: UTC sec 1680074987, PPS usec [actual] 191832984, PPS usec [estimated] 191832983
                    parts = data.split(',')
                    gps_mark = int(parts[0].split(" ")[3])
                    pps_meas = int(parts[1].split(" ")[4]) / 1.0e6
                    pps_est = int(parts[2].split(" ")[4]) / 1.0e6
                    pps_drift_ms = (pps_meas - pps_est) / 1000.0

                    gps_mark_list.append(gps_mark)
                    pps_meas_list.append(pps_meas)
                    pps_est_list.append(pps_est)
                    pps_drift_list.append(pps_drift_ms)
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting timing info from line: '{line['rawmsg'].strip()}' - {e}")
        return gps_mark_list, pps_meas_list, pps_est_list, pps_drift_list

    @staticmethod
    def extract_eof(session: list[dict]) -> tuple[list[float], list[int], list[float], list[float]]:
        """
        Extracts End-Of-File (EOF) information, including estimated file end time,
        block sequence numbers, and precise half-block and full-block timestamps.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[float], list[int], list[float], list[float]]: Four lists
            containing estimated EOF system times, block sequence numbers,
            half-block system timestamps, and full-block system timestamps, respectively.
        """
        eof_estimated: list[float] = []
        blk_seq: list[int] = []
        h_systime: list[float] = []
        f_systime: list[float] = []
        for line in session:
            if line['rawmsg'].startswith('EOF:'):
                try:
                    data = line['rawmsg']
                    # Example: EOF: 20312723534 BLK 211249 H 20312735340 F 20312639339
                    fields = data.split(' ')
                    e = int(fields[1]) / 1.0e6
                    b = int(fields[3])
                    h = int(fields[6]) / 1.0e6
                    f = int(fields[8]) / 1.0e6

                    eof_estimated.append(e)
                    blk_seq.append(b)
                    h_systime.append(h)
                    f_systime.append(f)
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting EOF info from line: '{line['rawmsg'].strip()}' - {e}")
        return eof_estimated, blk_seq, h_systime, f_systime

    @staticmethod
    def extract_temperatures(session: list[dict]) -> tuple[list[int], list[float]]:
        """
        Extracts temperature readings and their corresponding system timestamps
        from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[int], list[float]]: Two lists containing temperature values
            and their system timestamps.
        """
        temp_profile: list[int] = []
        sys_times: list[float] = []
        for line in session:
            if line['rawmsg'].startswith('ADC'):
                try:
                    data = line['rawmsg']
                    fields = data.split(" ")
                    # Assuming 'ADC' line format where temperature is the 3rd field
                    # ADC T 15347 Vref 24117 Vbatt(raw) 27830 Vbatt 6.465419 (V)
                    temp_field = int(fields[2])
                    temp_profile.append(temp_field)
                    sys_times.append(line['sysTime'])
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting temperature from line: '{line['rawmsg'].strip()}' - {e}")
        return temp_profile, sys_times

    @staticmethod
    def extract_voltages(session: list[dict]) -> tuple[list[float], list[float]]:
        """
        Extracts voltage readings and their corresponding system timestamps
        from a session's log messages.

        Args:
            session (list[dict]): A list of parsed log messages for a single session.

        Returns:
            tuple[list[float], list[float]]: Two lists containing voltage values
            and their system timestamps.
        """
        volt_profile: list[float] = []
        sys_times: list[float] = []
        for line in session:
            if line['rawmsg'].startswith('ADC'):
                try:
                    data = line['rawmsg']
                    fields = data.split(" ")
                    # Assuming 'ADC' line format where voltage is the 9th field
                    volt_field = float(fields[8])
                    volt_profile.append(volt_field)
                    sys_times.append(line['sysTime'])
                except (ValueError, IndexError) as e:
                    print(f"WARN: Error extracting voltage from line: '{line['rawmsg'].strip()}' - {e}")
        return volt_profile, sys_times