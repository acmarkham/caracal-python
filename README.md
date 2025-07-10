# caracal-python

Python libraries for ingesting, querying and returning data from Caracal acoustic sensors.

## Installation

1. Make a fresh conda environment (here we call it caracaldev):
`conda create -n caracaldev python=3.11`

2. Activate the conda environment:
`conda activate caracaldev`

3. Install the package and the requirements
`pip install git+https://github.com/acmarkham/caracal-python.git`

4. Import and use:
`import caracal`

## First steps: building an inventory

An inventory is a list of all the audio files and where they can be found. This allows for easy querying by date/time and location (GPS lat/lon from the Caracal itself). GPS timing information is also included in the inventory, as well as additional sensor parameters such as battery voltage (where available) and temperature.

1. Dump all the caracal data from SD cards into separate folders under a common root folder e.g. `Z:\Caracal`. 
2. Build the inventory - this crawls through all the folders, finds the `syslog.txt` files
```python
import caracal
rootpath = r"Z:\Caracal"
inv = caracal.inventorybuilder.InventoryBuilder()
inv.build(rootpath)
inv.save_inventory()
```
Once completed, it will save an `inventory.pkl` file under the rootpath. This can then be used later by the datagetter.

## Retrieving data

To retrieve data, we use the datagetter. We just point it at the rootpath where the inventory has been built and then can run location/time based queries e.g. to get all the audio data between 07:05 and 08:25 in a 1km radius around a query point. It will use the inventory to quickly find the matching audio files. These are returned with additional header/meta data (from the inventory) such as the location of the audio file. By default, audio is returned in mono format, but there is also an option to retrieve the data in quadrophonic (four channel) format for beamforming/direction-of-arrival estimation.

```python
import caracal
from datetime import datetime
rootpath = r"Z:\Caracal"
aGetter = caracal.DataGetter(rootpath=topDir)
response = aGetter.load_from_latlon(lat=lat,
                            lon=lon,
                            radius=1000,
                            start_time=datetime(2024, 1, 1, 7, 5, 0),
                            end_time=datetime(2024, 1, 1, 7, 5, 0))
print("Number of stations:", len(response.stations))
```



