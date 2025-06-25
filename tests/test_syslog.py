import unittest
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
import sys


# Assume the SyslogParser and dataclasses are in a file named syslog_parser.py
# If not, you might need to adjust the import path or put them in the same file.
from caracal.syslogparser import (
    Identity, PPS, Stats, Header, GPSlocation, AudioTiming, AudioFile, Session,
    SyslogContainer, SyslogParser
)

class TestSyslogParser(unittest.TestCase):

    def setUp(self):
        """
        Set up a temporary directory and a mock syslog file for testing.
        """
        self.test_dir = tempfile.mkdtemp()
        self.syslog_filename = os.path.join(self.test_dir, "syslog.txt")
        self.parser = SyslogParser(self.syslog_filename)

        # A realistic mock syslog content covering various log types and sessions
        # Using fixed timestamps for reproducibility and easier assertion
        self.mock_syslog_content = """
SYS,1,1000000,DEVICE_ID: 00112233 AABBCCDD EEFF0011
SYS,2,2000000,CARD_ID: ABCDEF12
SYS,3,3000000,Flash CRC: GHIJ3456
SYS,4,4000000,GPS_NAV2,fixType:3,lat:407127530,lon:-740059730,alt:100000,hAcc:1000,vAcc:2000,sAcc:500
SYS,5,5000000,GPS: UTC sec 1678886400, PPS usec [actual] 5000000, PPS usec [estimated] 5000000
SYS,6,6000000,Newfile: aud1678886400/file_1678886400.wav
SYS,7,7000000,ADC Temp:25.0C,Volt:3.7V,Bat:4.0V,Other:1.2V
SYS,8,8000000,EOF: 8000000 BLK 153600 H 8000000 F 8000000
SYS,9,9000000,GPS: UTC sec 1678886401, PPS usec [actual] 9000000, PPS usec [estimated] 9000000
SYS,10,10000000,Newfile: aud1678890000/file_1678890000.wav
SYS,11,11000000,EOF: 11000000 BLK 307200 H 11000000 F 11000000
SYS,12,12000000,ADC Temp:26.0C,Volt:3.8V,Bat:4.1V,Other:1.3V
SYS,13,13000000,GPS_NAV2,fixType:2,lat:407128000,lon:-740060000,alt:100000,hAcc:1000,vAcc:2000,sAcc:500
SYS,14,14000000,GPS: UTC sec 1678886402, PPS usec [actual] 14000000, PPS usec [estimated] 14000000
SYS,15,15000000,ADC Temp:27.0C,Volt:3.9V,Bat:4.2V,Other:1.4V
SYS,1,16000000,RESET_EVENT: Device restarted
SYS,2,17000000,DEVICE_ID: 00112233 AABBCCDD EEFF0011
SYS,3,18000000,CARD_ID: ABCDEF12
SYS,4,19000000,Flash CRC: GHIJ3456
SYS,5,20000000,GPS_NAV2,fixType:3,lat:407130000,lon:-740070000,alt:100000,hAcc:1000,vAcc:2000,sAcc:500
SYS,6,21000000,GPS: UTC sec 1678886403, PPS usec [actual] 21000000, PPS usec [estimated] 21000000
SYS,7,22000000,Newfile: aud1678893600/file_1678893600.wav
SYS,8,23000000,EOF: 23000000 BLK 460800 H 23000000 F 23000000
SYS,9,24000000,ADC Temp:28.0C,Volt:4.0V,Bat:4.3V,Other:1.5V
"""
        with open(self.syslog_filename, "w") as f:
            f.write(self.mock_syslog_content.strip())

    def tearDown(self):
        """
        Clean up the temporary directory after tests.
        """
        shutil.rmtree(self.test_dir)

    def test_parse_line(self):
        """
        Test the __parse_line static method.
        """
        # Valid line
        line = "SYS,1,1234567,Test message content"
        parsed = SyslogParser._SyslogParser__parse_line(line)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['msgCount'], 1)
        self.assertAlmostEqual(parsed['sysTime'], 1.234567)
        self.assertEqual(parsed['rawmsg'], "Test message content")

        # Invalid line (doesn't start with SYS)
        line = "LOG,1,1234567,Another message"
        parsed = SyslogParser._SyslogParser__parse_line(line)
        self.assertIsNone(parsed)

        # Line with missing fields
        line = "SYS,1,1234567"
        parsed = SyslogParser._SyslogParser__parse_line(line)
        self.assertIsNone(parsed) # Expect None due to IndexError in split

        # Line with non-integer msgCount or sysTime
        line = "SYS,A,1234567,Test"
        parsed = SyslogParser._SyslogParser__parse_line(line)
        self.assertIsNone(parsed) # Expect None due to ValueError

    def test_parse_syslog(self):
        """
        Test the __parse_syslog method for session separation.
        """
        sessions = self.parser._SyslogParser__parse_syslog(self.syslog_filename)
        self.assertIsNotNone(sessions)
        self.assertEqual(len(sessions), 2) # Expect two sessions based on mock content

        # Verify content of the first session
        self.assertEqual(len(sessions[0]), 15) # 15 lines in the first session
        self.assertEqual(sessions[0][0]['msgCount'], 1)
        self.assertEqual(sessions[0][0]['sysTime'], 1.0)
        self.assertEqual(sessions[0][-1]['msgCount'], 15)
        self.assertEqual(sessions[0][-1]['sysTime'], 15.0)

        # Verify content of the second session
        self.assertEqual(len(sessions[1]), 9) # 9 lines in the second session
        self.assertEqual(sessions[1][0]['msgCount'], 1)
        self.assertEqual(sessions[1][0]['sysTime'], 16.0)
        self.assertEqual(sessions[1][-1]['msgCount'], 9)
        self.assertEqual(sessions[1][-1]['sysTime'], 24.0)

        # Test empty file
        empty_filename = os.path.join(self.test_dir, "empty.txt")
        with open(empty_filename, "w") as f:
            f.write("")
        empty_sessions = self.parser._SyslogParser__parse_syslog(empty_filename)
        self.assertIsNone(empty_sessions)

        # Test file not found
        non_existent_filename = os.path.join(self.test_dir, "non_existent.txt")
        non_existent_sessions = self.parser._SyslogParser__parse_syslog(non_existent_filename)
        self.assertIsNone(non_existent_sessions)

    def test_process_basic_structure(self):
        """
        Test the overall structure of the SyslogContainer and Sessions after processing.
        """
        container = self.parser.process()
        self.assertIsNotNone(container)
        self.assertIsInstance(container, SyslogContainer)
        self.assertEqual(container.path, self.test_dir)
        self.assertEqual(container.filename, "syslog.txt")
        self.assertEqual(container.parserVersion, SyslogParser.PARSER_VERSION)

        self.assertEqual(len(container.sessions), 2)
        
        # Test first session
        session1 = container.sessions[0]
        self.assertIsInstance(session1, Session)
        self.assertEqual(session1.sessionID, 0)
        self.assertEqual(session1.path, self.test_dir)
        self.assertAlmostEqual(session1.header.sysDuration, 14.0) # 15.0 - 1.0

        # Test second session
        session2 = container.sessions[1]
        self.assertIsInstance(session2, Session)
        self.assertEqual(session2.sessionID, 1)
        self.assertAlmostEqual(session2.header.sysDuration, 8.0) # 24.0 - 16.0

    def test_process_identity_extraction(self):
        """
        Test identity extraction within the process method.
        """
        container = self.parser.process()
        
        # Session 1 Identity
        session1 = container.sessions[0]
        self.assertEqual(session1.header.headerID.cardID, "ABCDEF12")
        self.assertEqual(session1.header.headerID.deviceID, "00112233AABBCCDDEEFF0011")
        self.assertEqual(session1.header.headerID.versionHash, "GHIJ3456")

        # Session 2 Identity (should be same as mock content specifies)
        session2 = container.sessions[1]
        self.assertEqual(session2.header.headerID.cardID, "ABCDEF12")
        self.assertEqual(session2.header.headerID.deviceID, "00112233AABBCCDDEEFF0011")
        self.assertEqual(session2.header.headerID.versionHash, "GHIJ3456")

    def test_process_stats_extraction(self):
        """
        Test stats extraction within the process method.
        """
        container = self.parser.process()
        
        # Session 1 Stats
        session1_stats = container.sessions[0].header.stats
        self.assertEqual(session1_stats.num_msg, 15)
        self.assertEqual(session1_stats.num_files, 2)
        self.assertEqual(session1_stats.num_fixes, 2) # Two GPS_NAV2 lines
        self.assertEqual(session1_stats.num_timing, 3) # Three GPS: UTC sec lines

        # Median GPS (approximate based on mock data)
        # lat: 407127530 -> 40.7127530, 407128000 -> 40.7128000
        # lon: -740059730 -> -74.0059730, -740060000 -> -74.0060000
        self.assertAlmostEqual(session1_stats.median_GPS_lat, 40.7127765, places=7) # Median of 40.7127530, 40.7128000
        self.assertAlmostEqual(session1_stats.median_GPS_lon, -74.0059865, places=7) # Median of -74.0059730, -74.0060000

        # PPS Stats for Session 1
        self.assertAlmostEqual(session1_stats.ppsFirst.UTCTime, 1678886400)
        self.assertAlmostEqual(session1_stats.ppsFirst.sysTime, 5.0)
        self.assertAlmostEqual(session1_stats.ppsMid.UTCTime, 1678886401)
        self.assertAlmostEqual(session1_stats.ppsMid.sysTime, 9.0)
        self.assertAlmostEqual(session1_stats.ppsLast.UTCTime, 1678886402)
        self.assertAlmostEqual(session1_stats.ppsLast.sysTime, 14.0)

        # Voltage Stats for Session 1 (based on ADC lines 7, 12, 15)
        # ADC lines are SYS,7,7000000,ADC Temp:25.0C,Volt:3.7V,Bat:4.0V,Other:1.2V
        # SYS,12,12000000,ADC Temp:26.0C,Volt:3.8V,Bat:4.1V,Other:1.3V
        # SYS,15,15000000,ADC Temp:27.0C,Volt:3.9V,Bat:4.2V,Other:1.4V
        # Only 3 voltage readings, so mean of first 3 and last 3 will be the same.
        self.assertAlmostEqual(session1_stats.start_voltage, 3.8, places=2) # mean of [3.7, 3.8, 3.9]
        self.assertAlmostEqual(session1_stats.end_voltage, 3.8, places=2) # mean of [3.7, 3.8, 3.9]

        # Session 2 Stats
        session2_stats = container.sessions[1].header.stats
        self.assertEqual(session2_stats.num_msg, 9)
        self.assertEqual(session2_stats.num_files, 1)
        self.assertEqual(session2_stats.num_fixes, 1)
        self.assertEqual(session2_stats.num_timing, 1)

        self.assertAlmostEqual(session2_stats.median_GPS_lat, 40.7130000, places=7)
        self.assertAlmostEqual(session2_stats.median_GPS_lon, -74.0070000, places=7)

        # PPS Stats for Session 2
        self.assertAlmostEqual(session2_stats.ppsFirst.UTCTime, 1678886403)
        self.assertAlmostEqual(session2_stats.ppsFirst.sysTime, 21.0)
        self.assertAlmostEqual(session2_stats.ppsMid.UTCTime, 1678886403)
        self.assertAlmostEqual(session2_stats.ppsMid.sysTime, 21.0)
        self.assertAlmostEqual(session2_stats.ppsLast.UTCTime, 1678886403)
        self.assertAlmostEqual(session2_stats.ppsLast.sysTime, 21.0)

        # Voltage Stats for Session 2
        self.assertAlmostEqual(session2_stats.start_voltage, 4.0, places=2)
        self.assertAlmostEqual(session2_stats.end_voltage, 4.0, places=2)

    def test_process_audio_files_and_timing(self):
        """
        Test audio file and detailed timing extraction within the process method.
        """
        container = self.parser.process(detailed_PPS=True)
        session1 = container.sessions[0]

        self.assertEqual(len(session1.audioFiles), 2)

        # Test first audio file
        audio_file1 = session1.audioFiles[0]
        self.assertEqual(audio_file1.subpath, "aud1678886400/file_1678886400.wav")
        self.assertEqual(audio_file1.duration, SyslogParser.DURATION)
        self.assertEqual(audio_file1.utcTime, 1678886400.0)
        self.assertAlmostEqual(audio_file1.creation_sysTime, 6.0)

        # Timing for first audio file
        timing1 = audio_file1.timing
        self.assertAlmostEqual(timing1.ppsBefore.UTCTime, 1678886400) # GPS at 5.0 sysTime
        self.assertAlmostEqual(timing1.ppsBefore.sysTime, 5.0)
        self.assertAlmostEqual(timing1.ppsAfter.UTCTime, 1678886401) # GPS at 9.0 sysTime
        self.assertAlmostEqual(timing1.ppsAfter.sysTime, 9.0)
        self.assertAlmostEqual(timing1.ppsDuring.UTCTime, 1678886401) # Midpoint between 5.0 and 9.0
        self.assertAlmostEqual(timing1.ppsDuring.sysTime, 9.0)
        
        # EOF timing for first audio file (SYS,8,8000000,EOF: 8000000 BLK 153600 H 8000000 F 8000000)
        # Expected from previous EOF (if any) and current file creation time
        # This is the first EOF in the session, so a.timing.estimatedEndSystime, numBlocks, numSamples, blockLen are taken from the first EOF entry itself
        self.assertAlmostEqual(timing1.estimatedEndSystime, 8.0)
        self.assertAlmostEqual(timing1.blockStartSysTime, 8.0)
        self.assertAlmostEqual(timing1.blockEndSysTime, 8.0)
        self.assertEqual(timing1.numBlocks, 153600) # Current BLK from EOF line
        self.assertEqual(timing1.blockLen, SyslogParser.SAMPLES_PER_BLOCK)
        self.assertEqual(timing1.numSamples, 153600 * SyslogParser.SAMPLES_PER_BLOCK)


        # Test second audio file
        audio_file2 = session1.audioFiles[1]
        self.assertEqual(audio_file2.subpath, "aud1678890000/file_1678890000.wav")
        self.assertEqual(audio_file2.duration, SyslogParser.DURATION)
        self.assertEqual(audio_file2.utcTime, 1678890000.0)
        self.assertAlmostEqual(audio_file2.creation_sysTime, 10.0)

        # Timing for second audio file
        timing2 = audio_file2.timing
        self.assertAlmostEqual(timing2.ppsBefore.UTCTime, 1678886401) # GPS at 9.0 sysTime
        self.assertAlmostEqual(timing2.ppsBefore.sysTime, 9.0)
        self.assertAlmostEqual(timing2.ppsAfter.UTCTime, 1678886402) # GPS at 14.0 sysTime
        self.assertAlmostEqual(timing2.ppsAfter.sysTime, 14.0)
        self.assertAlmostEqual(timing2.ppsDuring.UTCTime, 1678886401) # Midpoint between 9.0 and 14.0
        self.assertAlmostEqual(timing2.ppsDuring.sysTime, 9.0)

        # EOF timing for second audio file (SYS,11,11000000,EOF: 11000000 BLK 307200 H 11000000 F 11000000)
        # Previous EOF: SYS,8,8000000,EOF: 8000000 BLK 153600 H 8000000 F 8000000
        # `creation_sysTime` for audio_file2 is 10.0. `searchsorted` for `eof_estimated` finds 11.0 (index 1)
        # So `eof_idx_before` will be 0 (for the 8.0 EOF), `eof_idx_after` will be 1 (for the 11.0 EOF)
        self.assertAlmostEqual(timing2.estimatedEndSystime, 11.0)
        self.assertAlmostEqual(timing2.blockStartSysTime, 8.0) # max(H,F) from previous EOF
        self.assertAlmostEqual(timing2.blockEndSysTime, 11.0) # max(H,F) from current EOF
        self.assertEqual(timing2.numBlocks, 307200 - 153600) # Current BLK - Previous BLK
        self.assertEqual(timing2.blockLen, SyslogParser.SAMPLES_PER_BLOCK)
        self.assertEqual(timing2.numSamples, (307200 - 153600) * SyslogParser.SAMPLES_PER_BLOCK)

    def test_process_detailed_pps(self):
        """
        Test that detailed PPS points are collected when detailed_PPS is True.
        """
        container = self.parser.process(detailed_PPS=True)
        session1 = container.sessions[0]
        session2 = container.sessions[1]

        # Session 1 should have 3 PPS points
        self.assertEqual(len(session1.header.PPSpoints), 3)
        self.assertIsInstance(session1.header.PPSpoints[0], PPS)
        self.assertAlmostEqual(session1.header.PPSpoints[0].UTCTime, 1678886400)
        self.assertAlmostEqual(session1.header.PPSpoints[0].sysTime, 5.0)

        # Session 2 should have 1 PPS point
        self.assertEqual(len(session2.header.PPSpoints), 1)
        self.assertAlmostEqual(session2.header.PPSpoints[0].UTCTime, 1678886403)
        self.assertAlmostEqual(session2.header.PPSpoints[0].sysTime, 21.0)

    def test_extract_node_id(self):
        """
        Test the static method extract_node_id.
        """
        mock_session = [
            {'rawmsg': 'DEVICE_ID: 00112233 AABBCCDD EEFF0011'},
            {'rawmsg': 'Some other log'},
        ]
        self.assertEqual(SyslogParser.extract_node_id(mock_session), "00112233AABBCCDDEEFF0011")

        mock_session_no_id = [{'rawmsg': 'Some other log'}]
        self.assertIsNone(SyslogParser.extract_node_id(mock_session_no_id))

    def test_extract_card_id(self):
        """
        Test the static method extract_card_id.
        """
        mock_session = [
            {'rawmsg': 'CARD_ID: ABCDEF12'},
            {'rawmsg': 'Some other log'},
        ]
        self.assertEqual(SyslogParser.extract_card_id(mock_session), "ABCDEF12")

        mock_session_no_id = [{'rawmsg': 'Some other log'}]
        self.assertIsNone(SyslogParser.extract_card_id(mock_session_no_id))

    def test_extract_firmware_hash(self):
        """
        Test the static method extract_firmware_hash.
        """
        mock_session = [
            {'rawmsg': 'Flash CRC: GHIJ3456'},
            {'rawmsg': 'Some other log'},
        ]
        self.assertEqual(SyslogParser.extract_firmware_hash(mock_session), "GHIJ3456")

        mock_session_no_hash = [{'rawmsg': 'Some other log'}]
        self.assertIsNone(SyslogParser.extract_firmware_hash(mock_session_no_hash))

    def test_extract_positions(self):
        """
        Test the static method extract_positions.
        """
        mock_session = [
            {'rawmsg': 'GPS_NAV2,fixType:3,lat:407127530,lon:-740059730,alt:100000,hAcc:1000,vAcc:2000,sAcc:500', 'sysTime': 10.0},
            {'rawmsg': 'GPS_NAV2,fixType:1,lat:100000000,lon:200000000', 'sysTime': 20.0}, # fixType 1 should be ignored
            {'rawmsg': 'GPS_NAV2,fixType:2,lat:300000000,lon:400000000', 'sysTime': 30.0},
            {'rawmsg': 'Not a GPS line', 'sysTime': 40.0}
        ]
        lons, lats, sys_times = SyslogParser.extract_positions(mock_session)
        self.assertEqual(len(lons), 2)
        self.assertEqual(len(lats), 2)
        self.assertEqual(len(sys_times), 2)
        self.assertAlmostEqual(lons[0], -74.0059730)
        self.assertAlmostEqual(lats[0], 40.7127530)
        self.assertAlmostEqual(sys_times[0], 10.0)
        self.assertAlmostEqual(lons[1], 40.0) # 400000000 / 1e7
        self.assertAlmostEqual(lats[1], 30.0) # 300000000 / 1e7
        self.assertAlmostEqual(sys_times[1], 30.0)

    def test_extract_files(self):
        """
        Test the static method extract_files.
        """
        mock_session = [
            {'rawmsg': 'Newfile: aud123/file_123.wav', 'sysTime': 100.0},
            {'rawmsg': 'Some other log', 'sysTime': 101.0},
            {'rawmsg': 'Newfile: aud456/file_456.wav', 'sysTime': 200.0},
        ]
        filenames, systimes = SyslogParser.extract_files(mock_session)
        self.assertEqual(len(filenames), 2)
        self.assertEqual(filenames[0], "aud123/file_123.wav")
        self.assertAlmostEqual(systimes[0], 100.0)
        self.assertEqual(filenames[1], "aud456/file_456.wav")
        self.assertAlmostEqual(systimes[1], 200.0)

    def test_extract_timing(self):
        """
        Test the static method extract_timing.
        """
        mock_session = [
            {'rawmsg': 'GPS: UTC sec 1678886400, PPS usec [actual] 5000000, PPS usec [estimated] 5000000', 'sysTime': 5.0},
            {'rawmsg': 'Some other log'},
            {'rawmsg': 'GPS: UTC sec 1678886401, PPS usec [actual] 9000000, PPS usec [estimated] 9000001', 'sysTime': 9.0},
        ]
        gps_marks, pps_meas, pps_est, pps_drift = SyslogParser.extract_timing(mock_session)
        self.assertEqual(len(gps_marks), 2)
        self.assertAlmostEqual(gps_marks[0], 1678886400)
        self.assertAlmostEqual(pps_meas[0], 5.0)
        self.assertAlmostEqual(pps_est[0], 5.0)
        self.assertAlmostEqual(pps_drift[0], 0.0)
        
        self.assertAlmostEqual(gps_marks[1], 1678886401)
        self.assertAlmostEqual(pps_meas[1], 9.0)
        self.assertAlmostEqual(pps_est[1], 9.000001)
        self.assertAlmostEqual(pps_drift[1], -0.000001) # (9.0 - 9.000001) / 1000

    def test_extract_eof(self):
        """
        Test the static method extract_eof.
        """
        mock_session = [
            {'rawmsg': 'EOF: 10000000 BLK 100 H 10000000 F 9999000', 'sysTime': 10.0},
            {'rawmsg': 'Some other log'},
            {'rawmsg': 'EOF: 20000000 BLK 200 H 20000000 F 20001000', 'sysTime': 20.0},
        ]
        eof_est, blk_seq, h_sys, f_sys = SyslogParser.extract_eof(mock_session)
        self.assertEqual(len(eof_est), 2)
        self.assertAlmostEqual(eof_est[0], 10.0)
        self.assertEqual(blk_seq[0], 100)
        self.assertAlmostEqual(h_sys[0], 10.0)
        self.assertAlmostEqual(f_sys[0], 9.999)

        self.assertAlmostEqual(eof_est[1], 20.0)
        self.assertEqual(blk_seq[1], 200)
        self.assertAlmostEqual(h_sys[1], 20.0)
        self.assertAlmostEqual(f_sys[1], 20.001)

    def test_extract_temperatures(self):
        """
        Test the static method extract_temperatures.
        """
        mock_session = [
            {'rawmsg': 'ADC Temp:25.0C,Volt:3.7V,Bat:4.0V,Other:1.2V', 'sysTime': 50.0},
            {'rawmsg': 'Some other log'},
            {'rawmsg': 'ADC Temp:28.5C,Volt:3.8V,Bat:4.1V,Other:1.3V', 'sysTime': 60.0},
        ]
        temps, sys_times = SyslogParser.extract_temperatures(mock_session)
        self.assertEqual(len(temps), 2)
        self.assertEqual(temps[0], 25)
        self.assertAlmostEqual(sys_times[0], 50.0)
        self.assertEqual(temps[1], 28) # ADC Temp:28.5C, int() conversion
        self.assertAlmostEqual(sys_times[1], 60.0)

    def test_extract_voltages(self):
        """
        Test the static method extract_voltages.
        """
        mock_session = [
            {'rawmsg': 'ADC Temp:25.0C,Volt:3.7V,Bat:4.0V,Other:1.2V', 'sysTime': 50.0},
            {'rawmsg': 'Some other log'},
            {'rawmsg': 'ADC Temp:28.5C,Volt:3.8V,Bat:4.1V,Other:1.3V', 'sysTime': 60.0},
        ]
        volts, sys_times = SyslogParser.extract_voltages(mock_session)
        self.assertEqual(len(volts), 2)
        self.assertAlmostEqual(volts[0], 3.7)
        self.assertAlmostEqual(sys_times[0], 50.0)
        self.assertAlmostEqual(volts[1], 3.8)
        self.assertAlmostEqual(sys_times[1], 60.0)

    def test_empty_syslog_file_process(self):
        """
        Test processing an empty syslog file.
        """
        empty_filename = os.path.join(self.test_dir, "empty.txt")
        with open(empty_filename, "w") as f:
            f.write("")
        
        parser = SyslogParser(empty_filename)
        container = parser.process()
        self.assertIsNone(container)

    def test_syslog_file_not_found_process(self):
        """
        Test processing a non-existent syslog file.
        """
        non_existent_filename = os.path.join(self.test_dir, "non_existent.txt")
        parser = SyslogParser(non_existent_filename)
        container = parser.process()
        self.assertIsNone(container)

    def test_single_session_syslog_process(self):
        """
        Test processing a syslog with only a single session.
        """
        single_session_content = """
SYS,1,1000000,DEVICE_ID: SINGLE_DEVICE
SYS,2,2000000,CARD_ID: SINGLE_CARD
SYS,3,3000000,Newfile: aud1/file_single.wav
SYS,4,4000000,EOF: 4000000 BLK 100 H 4000000 F 4000000
"""
        single_session_filename = os.path.join(self.test_dir, "single_session.txt")
        with open(single_session_filename, "w") as f:
            f.write(single_session_content.strip())
        
        parser = SyslogParser(single_session_filename)
        container = parser.process()
        self.assertIsNotNone(container)
        self.assertEqual(len(container.sessions), 1)
        
        session = container.sessions[0]
        self.assertEqual(session.sessionID, 0)
        self.assertEqual(session.header.headerID.deviceID, "SINGLE_DEVICE")
        self.assertEqual(len(session.audioFiles), 1)
        self.assertEqual(session.audioFiles[0].subpath, "aud1/file_single.wav")
        self.assertAlmostEqual(session.audioFiles[0].creation_sysTime, 3.0)

    def test_robustness_missing_data_lines(self):
        """
        Test robustness when expected data lines (GPS, PPS, EOF) are missing.
        """
        sparse_syslog_content = """
SYS,1,1000000,DEVICE_ID: SPARSE_DEV
SYS,2,2000000,CARD_ID: SPARSE_CARD
SYS,3,3000000,Newfile: aud_sparse/file_sparse.wav
SYS,4,4000000,ADC Temp:20C,Volt:3.5V
"""
        sparse_filename = os.path.join(self.test_dir, "sparse.txt")
        with open(sparse_filename, "w") as f:
            f.write(sparse_syslog_content.strip())
        
        parser = SyslogParser(sparse_filename)
        container = parser.process()
        self.assertIsNotNone(container)
        self.assertEqual(len(container.sessions), 1)
        session = container.sessions[0]

        # Check stats for missing data
        self.assertEqual(session.header.stats.num_fixes, 0)
        self.assertEqual(session.header.stats.num_timing, 0)
        self.assertAlmostEqual(session.header.stats.median_GPS_lat, 0.0)
        self.assertAlmostEqual(session.header.stats.median_GPS_lon, 0.0)
        
        # PPS and EOF timing should be default_factory values (0 or empty lists)
        self.assertAlmostEqual(session.header.stats.ppsFirst.UTCTime, 0.0)
        self.assertEqual(len(session.header.PPSpoints), 0)
        
        audio_file = session.audioFiles[0]
        self.assertAlmostEqual(audio_file.timing.ppsBefore.UTCTime, 0.0)
        self.assertAlmostEqual(audio_file.timing.estimatedEndSystime, 0.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
