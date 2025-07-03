# inventorybuilder.py
import glob
import os
import pickle # Import pickle for saving/loading inventory
import datetime # Import datetime for timestamping the inventory
from dataclasses import dataclass, field # Import dataclass and field
from tqdm import tqdm # Import tqdm for progress bar

from .syslogparser import SyslogParser, SyslogContainer # Explicitly import SyslogContainer for type hinting

@dataclass
class CaracalInventory:
    """
    A dataclass to hold the built inventory of syslog data along with metadata.

    Attributes:
        syslog_containers (list[SyslogContainer]): A list of parsed SyslogContainer objects.
        build_date (datetime.datetime): The date and time when the inventory was built.
        base_path (str): The root directory from which the inventory was built.
        comments (str): Any additional comments or notes about this inventory.
    """
    syslog_containers: list[SyslogContainer] = field(default_factory=list)
    build_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    base_path: str = ""
    comments: str = ""


class InventoryBuilder:
    """
    Builds an inventory of syslog data by crawling a specified root directory.
    It identifies syslog files, parses them, and stores the extracted information
    in a CaracalInventory dataclass.
    """

    def __init__(self):
        """
        Initializes the InventoryBuilder.
        """
        self.rootdir: str | None = None
        self.inventory: CaracalInventory = CaracalInventory() # Initialize with an empty CaracalInventory

    def build(self, rootdir: str, comments: str = "") -> CaracalInventory:
        """
        Runs the data-crawling process to find and parse syslog files,
        building a data index encapsulated in a CaracalInventory object.
        The paths within the SyslogContainer objects are converted to be
        relative to the provided root directory for portability.

        Args:
            rootdir (str): The root directory path to explore for syslog files.
            comments (str, optional): Additional comments to store with the inventory.

        Returns:
            CaracalInventory: An instance of CaracalInventory containing
                              all parsed syslog data and metadata.
        """
        print(f"Starting the discovery build in: {rootdir}")
        self.rootdir = rootdir
        syslogs = self.__find_all_syslogs()
        print(f"Found {len(syslogs)} syslog files.")

        parsed_syslog_containers: list[SyslogContainer] = []
        # Wrap the loop with tqdm for a progress bar
        for syslog_path in tqdm(syslogs, desc="Parsing syslogs"):
            # print(f"Parsing: {syslog_path}") # Removed to avoid cluttering progress bar output
            parser = SyslogParser(syslog_path)
            syslog_container = parser.process()

            if syslog_container:
                relative_path = os.path.relpath(syslog_container.path, self.rootdir)
                syslog_container.path = relative_path
                for session in syslog_container.sessions:
                    session.path = relative_path
                parsed_syslog_containers.append(syslog_container)
            else:
                # print(f"WARNING: No valid data parsed from {syslog_path}. Skipping.") # Removed to avoid cluttering progress bar output
                pass # Keep pass to explicitly do nothing

        # Create the CaracalInventory instance
        self.inventory = CaracalInventory(
            syslog_containers=parsed_syslog_containers,
            build_date=datetime.datetime.now(),
            base_path=self.rootdir,
            comments=comments
        )
        return self.inventory

    def __find_all_syslogs(self) -> list[str]:
        """
        Internal method to find all syslog.txt files within the `rootdir`
        and its subdirectories.

        Returns:
            list[str]: A list of full paths to all found syslog files.
        """
        if self.rootdir is None:
            raise ValueError("rootdir must be set before finding syslogs. Call build() first.")

        print("Finding syslogs...")
        files = glob.glob(os.path.join(self.rootdir, '**', '*syslog*.txt'), recursive=True)
        return files

    def save_inventory(self, filename: str | None = None):
        """
        Saves the built inventory (CaracalInventory object) to a pickle file.
        If no filename is provided, it defaults to "inventory.pkl" in the
        base path used for building the inventory.

        Args:
            filename (str, optional): The path and filename where the inventory should be saved.
                                      If None, defaults to "inventory.pkl" in the root directory
                                      where the inventory was built.
        """
        if not self.inventory.syslog_containers: # Check if the inventory actually contains data
            print("WARNING: No inventory to save. Build the inventory first.")
            return

        if filename is None:
            if self.rootdir is None:
                print("ERROR: Cannot determine default save path. 'rootdir' is not set. Please call build() first or provide an explicit filename.")
                return
            filename_to_save = os.path.join(self.rootdir, "inventory.pkl")
        else:
            filename_to_save = filename

        try:
            with open(filename_to_save, 'wb') as f:
                pickle.dump(self.inventory, f)
            print(f"Inventory successfully saved to: {filename_to_save}")
        except Exception as e:
            print(f"ERROR: Failed to save inventory to {filename_to_save}: {e}")

    @staticmethod
    def load_inventory(filename: str) -> CaracalInventory | None:
        """
        Loads a CaracalInventory object from a pickle file.

        Args:
            filename (str): The path and filename of the pickled inventory file.

        Returns:
            CaracalInventory | None: The loaded CaracalInventory object, or None if loading fails.
        """
        try:
            with open(filename, 'rb') as f:
                inventory = pickle.load(f)
            if isinstance(inventory, CaracalInventory):
                print(f"Inventory successfully loaded from: {filename}")
                return inventory
            else:
                print(f"ERROR: Loaded object from {filename} is not a CaracalInventory instance.")
                return None
        except FileNotFoundError:
            print(f"ERROR: Inventory file not found: {filename}")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load inventory from {filename}: {e}")
            return None

