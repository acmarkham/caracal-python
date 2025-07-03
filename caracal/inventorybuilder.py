import glob
import os
from .syslogparser import SyslogParser

class InventoryBuilder:

    def __init__(self):
        """
        Setup the data-crawling process
        """
        pass


    def build(self,rootdir):
        """
        Run the data-crawling process and build the dataindex

        Keyword arguments:
        rootdir -- path (as string) to explore

        Returns:
        dataindex - a dataindex structure
        """
        print("Starting the discovery build in",rootdir)
        self.rootdir = rootdir
        syslogs = self.__find_all_syslogs()
        print("Found",len(syslogs))
        sys = []
        # parse all the syslogs
        for syslog in syslogs:
            print("Parsing",syslog)
            s = SyslogParser(syslog)
            syslogcontainer = s.process()
            # We now reach into the top level syslog and very naughtily change the full, absolute path
            # to a path relative to the rootdir. This makes it more portable e.g. if drive letters change.
            relative_path = os.path.relpath(syslogcontainer.path, self.rootdir)
            syslogcontainer.path = relative_path
            for session in syslogcontainer.sessions:
                session.path = relative_path
            sys.append(syslogcontainer)
        return sys
            


    def __find_all_syslogs(self):
        """ 
        Internal method to return a list of all syslogs in the rootdir
        """
        print("Finding syslogs..")
        files = glob.glob(self.rootdir + '/**/*syslog*.txt', recursive=True)
        return files
