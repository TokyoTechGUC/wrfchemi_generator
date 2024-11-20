# wrfchemi_generator
Python-based `wrfchemi` generator using EDGAR-HTAPv3 dataset.
Prepared by Alvin C.G. Varquez, GUC Lab, Institute of Science Tokyo.

### Installation
Download the following files in the folder. 
* `generate_wrfchempi.py`
* `alloc_racm_species.csv` - Extracted from Stockwell et al. "A new mechanism for regional atmospheric chemistry modeling", 1997
* `NMVOC_speciation_HTAP_v3.xls` - Extracted from NMVOC Speciation profiles [here](https://edgar.jrc.ec.europa.eu/dataset_htap_v3).

### Prerequisites
Ensure that the modules are installed.
Refer to `requirements.txt` or refer to the modules imported in `generate_wrfchempi.py`

### Necessary settings before running the program.
Edit the lines containing the following information in `generate_wrfchempi.py`
* `edgar_folder` - Path for downloading the EDGAR_HTAPV3 data
* `wrfinput_folder` - Path of `wrfinput_d*` files used as time and space reference.
* `edgar_year` - Reference year from the EDGAR data. The latest is 2018.
* `month` - Integer representation of the month.
* `country` - Path of the wrfinput file. Refer to the `NMVOC_speciation_HTAP_v3.xls` for the available countries. If country is not listed, use the nearest country.
* `reference_excel_htap` - Path of `NMVOC_speciation_HTAP_v3.xls`.
* `alloc_racm_reference` - Path of `alloc_racm_species.csv`.
* `out_folder` - Directory where the `wrfchem*` files will be stored.

### Disclaimer and Caution
This approach is currently under development and is not final.
The developers will not be held responsible for any errors encountered in the implementation.
