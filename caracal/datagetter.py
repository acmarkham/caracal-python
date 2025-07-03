import pickle
from dataclasses import dataclass,field
from .syslogparser import Identity,Stats,AudioFile,Header,Session
from .position import NamedLocation,NamedLocationLoader,OverrideLoader

# to load a wavfile:
from scipy.io import wavfile
import numpy
from zoneinfo import ZoneInfo
import datetime
import os
import soundfile as sf
import csv


# Container for multiple returns
@dataclass
class CaracalMultipleAudioData:
    stations:list # list of CaracalAudioData

    # save this wavefile container
    #def saveWav(self,filename):
    #    wavfile.write(filename=filename,
    #                  data=self.audioData,
    #                  rate=self.sampleRate)

# Container for single station's data
@dataclass
class CaracalAudioData:
    path = ''
    header:Header=field(default_factory=Header)
    audioFile:AudioFile=field(default_factory=AudioFile) 
    UTCstart:int=0
    UTCEnd:int=0
    sampleRate:int=0
    audioData = None
    locationMatch:str = 'Unspecified'
    stationName:str = 'Unknown'

    # save this wavefile container
    #def saveWav(self,filename):
    #    wavfile.write(filename=filename,
    #                  data=self.audioData,
    #                  rate=self.sampleRate)




class DataGetter:

    # Index of the mono channel in the wave file to use
    CH_MONO_IDX = 0
    # Decimal degrees to m conversion
    DEGREES_TO_METRES = 111319.5

    def __init__(self,
                 rootpath, 
                 syslogdata,
                 locationinfo=None,
                 timezone:str="UTC",
                 audio_mode='mono',
                 overrideinfo=None):
        '''
        
        Initialize a datagetter instance. 
        
        rootpath: The main, top level path where all the data lives. Typically, this will be the same
        path as used for datadiscovery.
        syslogdata: This must be supplied with a syslogdata (pickled list of syslogcontainers).
        locationinfo: An optional locationinfo handle can be passed in for semantic/surveyed locations
        timezone string (ZoneInfo code): Pass in an optional timezone string (defaults to UTC) to convert
        naive/unaware datetimes to the correct timestamps
        audio_mode: 'mono' or 'quad', default to mono
        overrideinfo: An option overrideinfo handle can be passed in to assist with semantic/surveyed locations'''

        self.rootpath = rootpath
        self.syslogdata = syslogdata
        self.locationinfo = locationinfo
        self.loc = None
        if audio_mode == 'mono':
            self.audio_mode = 'mono'
            print("DataGetter: Audio Mode set to Mono")
        elif audio_mode == 'quad':
            print("DataGetter: Audio Mode set to quad")
            self.audio_mode = 'quad'
        else:
            raise ValueError("Audio mode must be 'mono' or 'quad'")
        self.audio_mode = audio_mode

        # timezone
        self.timezone_str = timezone
        self.local_timezone = ZoneInfo(timezone)
        print("DataGetter: Timezone set to:",self.local_timezone)
        # Unpickle (load) the parsed syslogs to check
        with open(syslogdata, 'rb') as handle:
            self.sys = pickle.load(handle)
            print("DataGetter: Loaded:",len(self.sys),"syslog entries")
        if locationinfo is not None:
            self.loc = NamedLocationLoader(locationinfo)
        if overrideinfo is not None:
            self.override = OverrideLoader(overrideinfo)
        else:
            self.override = None

    def __convert(self,dt):
      # Check if timezone has been set. If not, assume that it is in the naive timezone and convert
      # to an aware date-time.
      if (dt.tzinfo is None):
         dt = dt.replace(tzinfo=self.local_timezone)    
      # First change to UTC if necessary
      utc_dt = dt.astimezone(ZoneInfo("UTC"))
      # Then convert to unix timestamps
      utc = utc_dt.timestamp()
      return utc
    
    @staticmethod
    def __unpack(dat):
        '''Internal function to inflate a packed wave file into four channels'''
        num_samples = numpy.shape(dat)[0]
        new_dat = numpy.empty((num_samples,4),dtype=numpy.int32)
        gain_array = numpy.empty(num_samples)
        dat_int = dat.astype(numpy.int32)
        for idx in numpy.arange(num_samples):
            d = dat_int[idx]
            ch_a = d[0]&0xFFFFFE00
            ch_b = (d[1]&0xFFF00000)>>20
            ch_c = (d[1]&0x000FFF00)>>8
            ch_d = (d[1]&0x000000FF)<<4 | (d[0]&0x0F)
            gain = (d[0]&0x1f0)>>4
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
    def load_wav(filename, start_offset=None,duration=None,audio_mode='mono'):
        '''Load audio, return as a numpy array and sample rate
        - filename: full path to .wav file to open
        - start_offset: [optional] offset in seconds (can be fractional) relative to start timestamp of file
        - duration: [optional] how many seconds of audio (can be fractional) to load
        
        Note: This is a fast method that does not load the whole file into
        memory. It is much quicker than pulling the whole wave file from a
        server. It instead just seeks the correct chunk. It requires the 
        soundfile library.
        '''
        #https://stackoverflow.com/questions/62957499/reading-only-a-part-of-a-large-wav-file-with-python
        #print("Using Soundfile Loader")
        
        track = sf.SoundFile(filename)

        can_seek = track.seekable() # True
        if not can_seek:
            raise ValueError("Not compatible with seeking")
        sr = track.samplerate
        if start_offset is not None:
            start_frame = int(start_offset*sr)
        else:
            start_frame = 0
        if duration is not None:
            frames_to_read = int(sr * (duration))
            track.seek(start_frame)
            if audio_mode == 'mono':
                audio_section = track.read(frames_to_read)
                return sr,numpy.squeeze(audio_section[:,DataGetter.CH_MONO_IDX])
            elif audio_mode == 'quad':
                # explicit int32 for bit manipulation
                audio_section = track.read(frames_to_read,dtype='int32')
                quad_audio, gain = DataGetter.__unpack(audio_section)
                # scaling back to float64 [-1,1] range
                audio_float = numpy.array(quad_audio).astype(numpy.float64)
                audio_scaled = quad_audio/2**31
                return sr,audio_scaled
            else:
                raise ValueError("Audio mode must be 'mono' or 'quad'")
        else:
            raise ValueError("Need a duration")


    @staticmethod
    def load_wav_legacy(filename,start_offset=None,duration=None):
        '''Load (mono) audio, return as a numpy array and sample rate
        - filename: full path to .wav file to open
        - start_offset: [optional] offset in seconds (can be fractional) relative to start timestamp of file
        - duration: [optional] how many seconds of audio (can be fractional) to load
        '''
        try:
            samplerate, data = wavfile.read(filename)
        except:
            return None
        if start_offset is not None:
            offset_idx = int(start_offset*samplerate)
        else:
            offset_idx = 0
        if duration is not None:
            end_idx = int(offset_idx + duration*samplerate)
            #print("1",offset_idx,end_idx)
            return samplerate,numpy.squeeze(data[offset_idx:end_idx,DataGetter.CH_MONO_IDX])
        else:
            #print("2",offset_idx)
            return samplerate,numpy.squeeze(data[offset_idx:,DataGetter.CH_MONO_IDX])


    def get_audio_from_session(self,session,audiofile,start_timestamp,end_timestamp):
        '''Returns a container (CaracalAudioData) with the requested audio data'''
        c = CaracalAudioData()
        c.path = session.path
        c.header = session.header
        c.audioFile = audiofile
        c.UTCstart = start_timestamp
        c.UTCEnd = end_timestamp
        # work out the offset and duration
        offset = c.UTCstart - c.audioFile.utcTime
        if offset < 0:
            print("ERR:negative starting index")
            offset = 0
        duration = c.UTCEnd-c.UTCstart
        try:
            full_filename = os.path.join(self.rootpath,
                                                    session.path,
                                                    audiofile.subpath)
            s,aud = DataGetter.load_wav(full_filename,
                                        offset,
                                        duration,
                                        audio_mode=self.audio_mode)
        except Exception as e:
            print("Failed to get audio from",full_filename,c.audioFile,e)
            aud = None
        # print("got",numpy.shape(aud))
        if aud is not None:
            #print("setting audioData",numpy.shape(aud))
            c.audioData = aud
            c.sampleRate = s
        return c



    def load_from_session(self,session,start_time,end_time):
        '''Get the (single) CaracalAudioData corresponding to parameters:
        - a session (of instance Session)
        - a start_time (UTC seconds or datetime instance)
        - an end_time (UTC seconds or datetime instance)
        
        - Returns:
        : None - no data or parameters don't work
        : instance of CaracalAudioData'''
        if isinstance(session,Session) is False:
            print("ERR:Not a session")
            return None
        # convert from datetime to timestamp
        if isinstance(start_time,datetime.date):
            start_time = self.__convert(start_time)
        # convert from datetime to timestamp
        if isinstance(end_time,datetime.date):
            end_time = self.__convert(end_time)

        for audiofile in session.audioFiles:
            if audiofile.utcTime <= start_time:
                if (audiofile.utcTime + audiofile.duration) >= end_time:
                    c = self.get_audio_from_session(session,
                                              audiofile,
                                              start_time,
                                              end_time)
                    return c
        # Failing condition, no return
        return None
                
    


    def load_from_device_id(self,devID,start_time,end_time):
        '''Load a file given:
        - a deviceID
        - a start_time (UTC seconds or datetime instance)
        - an end time (UTC seconds or datetime instance)
        
        - Returns:
        : None - no data or parameters don't work
        : instance of CaracalAudioData'''

        # convert from datetime to timestamp
        if isinstance(start_time,datetime.date):
            start_time = self.__convert(start_time)
        # convert from datetime to timestamp
        if isinstance(end_time,datetime.date):
            end_time = self.__convert(end_time)

        # Step 1: scan through the sys file to match all the sessions
        # that have the same deviceID
        for syslogcontainer in self.sys:
            for session in syslogcontainer.sessions:
                sessID = session.header.headerID.deviceID
                if devID == sessID:
                    c = self.load_from_session(session,
                                                   start_time,
                                                   end_time)
                    # Only exit if there was a timestamp match
                    if c is not None:
                        return c
        # Failing condition, no return
        return None   


    def load_from_latlon(self,
                                        lat,lon,
                                        start_time,
                                        end_time,
                                        location_threshold=20.0): 
        '''Load a file given:
        - a lat/lon location (decimal degrees)
        - a start_time (UTC seconds or datetime instance)
        - an end time (UTC seconds or datetime instance)
        - location_threshold (m) - [default:20.0 m]

        NB: this will return the first match it finds 
        (not necessarily the closest location match)

        NB: This is a rough approximation between decimal degrees and
        on-the-ground distance, accurate at the equator
        
        - Returns:
        : None - no data or parameters don't work
        : instance of CaracalAudioData''' 

        # convert from datetime to timestamp
        if isinstance(start_time,datetime.date):
            start_time = self.__convert(start_time)
        # convert from datetime to timestamp
        if isinstance(end_time,datetime.date):
            end_time = self.__convert(end_time)  

        # Step 1: scan through the sys file to match all the sessions
        # that have the same deviceID
        for syslogcontainer in self.sys:
            for session in syslogcontainer.sessions:
                median_lon = session.header.stats.median_GPS_lon
                median_lat = session.header.stats.median_GPS_lat
                delta_lon = median_lon-lon
                delta_lat = median_lat-lat
                delta_range_lon = delta_lon * DataGetter.DEGREES_TO_METRES
                delta_range_lat = delta_lat * DataGetter.DEGREES_TO_METRES
                total_range = numpy.sqrt(delta_range_lon**2+delta_range_lat**2)
                # print("Range",total_range)
                if total_range < location_threshold:
                    c = self.load_from_session(session,
                                                   start_time,
                                                   end_time)
                    if c is not None:
                        c.locationMatch = 'FromDevice'
                        return c
                    
        # Option Step 2: If over-ride is present, use this instead to enforce the matching
        if self.override is not None:
            for syslogcontainer in self.sys:
                for session in syslogcontainer.sessions:
                    sessionName = self.override.getNameFromPath(session.path) 
                    if sessionName is not None:   
                        surveyedPos = self.loc.fromName(sessionName) 
                        if surveyedPos is not None:
                            delta_lon = surveyedPos.lon-lon
                            delta_lat = surveyedPos.lat-lat
                            delta_range_lon = delta_lon * DataGetter.DEGREES_TO_METRES
                            delta_range_lat = delta_lat * DataGetter.DEGREES_TO_METRES
                            total_range = numpy.sqrt(delta_range_lon**2+delta_range_lat**2)
                            # print("Range",total_range)
                            if total_range < location_threshold:
                                c = self.load_from_session(session,
                                                            start_time,
                                                            end_time)
                                if c is not None:
                                    c.locationMatch = 'FromOverride'
                                    return c   
        # Failing condition, no return
        return None   

    def load_from_name(self,
                        name,
                        start_time,
                        end_time,
                        location_threshold=20.0): 
        '''Load a file given:
        - a name (e.g. 'M04')
        - a start_time (UTC seconds or datetime instance)
        - an end time (UTC seconds or datetime instance)
        - location_threshold (m) - [default:20.0 m]

        NB: this will return the first match it finds 
        (not necessarily the closest location match)
        
        - Returns:
        : None - no data or parameters don't work
        : instance of CaracalAudioData'''

        # convert from datetime to timestamp
        if isinstance(start_time,datetime.date):
            start_time = self.__convert(start_time)
        # convert from datetime to timestamp
        if isinstance(end_time,datetime.date):
            end_time = self.__convert(end_time)

        if self.loc is None:
            print ("ERR: no loc file")
            return None
        if len(self.loc.getAllNamedPos()) == 0:
            print ("ERR: no named locations")
            return None
        p = self.loc.fromName(name)
        if p is None:
            print("ERR: No matching named location")
            return None
        
        c = self.load_from_latlon(
            p.lat,
            p.lon,
            start_time,
            end_time,
            location_threshold)
        return c
    


    def load_around_latlon(self,
                                        lat,lon,
                                        start_time,
                                        end_time,
                                        radius): 
        '''Load and return *multiple* files given:
        - a lat/lon location (decimal degrees)
        - a start_timestamp (UTC seconds or datetime instance)
        - an end timestamp (UTC seconds or datetime instance)
        - query radius (m) 

        NB: This is a rough approximation between decimal degrees and
        on-the-ground distance, accurate at the equator
        
        - Returns:
        : None - no data or parameters don't work
        : List [instances of CaracalAudioData]'''   

        # convert from datetime to timestamp
        if isinstance(start_time,datetime.date):
            start_time = self.__convert(start_time)
        # convert from datetime to timestamp
        if isinstance(end_time,datetime.date):
            end_time = self.__convert(end_time)

        matching_instances = []

        for syslogcontainer in self.sys:
            for session in syslogcontainer.sessions:
                median_lon = session.header.stats.median_GPS_lon
                median_lat = session.header.stats.median_GPS_lat
                delta_lon = median_lon-lon
                delta_lat = median_lat-lat
                delta_range_lon = delta_lon * DataGetter.DEGREES_TO_METRES
                delta_range_lat = delta_lat * DataGetter.DEGREES_TO_METRES
                total_range = numpy.sqrt(delta_range_lon**2+delta_range_lat**2)
                # print("Range",total_range)
                if total_range < radius:
                    c = self.load_from_session(session,
                                                   start_time,
                                                   end_time)
                    if c is not None:
                        matching_instances.append(c)
        if len(matching_instances) == 0:
            return None
        return matching_instances 
    
    def load_around_name(self,
                                        name,
                                        start_time,
                                        end_time,
                                        location_threshold=20.0,
                                        radius=1000.0): 
        '''Load a file given:
        - a name (e.g. 'M04')
        - a start_time (UTC seconds or datetime instance)
        - an end time (UTC seconds or datetime instance)
        - location_threshold (m) - [default:20.0 m]
        - radius (m) - [default:1000.0m]

        
        - Returns:
        : None - no data or parameters don't work
        : List [instances of CaracalAudioData]'''

        if self.loc is None:
            print ("ERR: no loc file")
            return None
        if len(self.loc.getAllNamedPos()) == 0:
            print ("ERR: no named locations")
            return None
        p = self.loc.fromName(name)
        if p is None:
            print("ERR: No matching named location")
            return None
        c = self.load_around_latlon(
            p.lat,
            p.lon,
            start_time,
            end_time,
            radius)
        return c
    
    




        


    
