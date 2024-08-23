import xarray as xr
import subprocess
import os
import numpy as np
from glob import glob
import pandas as pd
from cdo import *
import xesmf as xe
import cf_xarray

cdo = Cdo()

compounds = ['BC','CO','NH3','NMVOC','NOx','OC','PM10','PM2.5','SO2']
edgar_folder = 'input_edgar'
wrfinput_folder = '/gs/bs/tga-guc-lab/users/dea/simulations/two_weeks/newmegan_urban_ahe/'
edgar_year = 2018
month =8 
country = 'Japan'
reference_excel_htap = './NMVOC_speciation_HTAP_v3.xls'
alloc_racm_reference = './alloc_racm_species.csv'
out_folder = './'

def download_raw_emissions():
    if not os.path.isdir(edgar_folder): os.mkdir(edgar_folder)
    for icomp in compounds:
        ofile = f'{edgar_folder}/edgar_HTAPv3_{edgar_year}_{icomp}.zip'
        check_file = f'{edgar_folder}/trimmed_edgar_HTAPv3_{edgar_year}_{icomp}.nc'
        print(check_file)
        url = f'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/htap_v3/gridmaps_01x01/emissions/{icomp}/{os.path.basename(ofile)}'
        if not os.path.exists(check_file):
            result = subprocess.run(['wget',url,'-O',ofile],capture_output=True,text=True)
            result = subprocess.run(['unzip','-o',ofile])
            result = subprocess.run(['mv',os.path.basename(ofile).replace('.zip','.nc'),f'./{edgar_folder}/'])
            result = subprocess.run(['rm',ofile])
        else:
            print(f"{ofile} already exists. Skipping download.")
    os.remove("_readme.html")
    return 0

def trim_to_wrfinput(df,clear_raw=False):
    min_lat = np.min(df['XLAT'].values)-0.25
    max_lat = np.max(df['XLAT'].values)+0.25
    min_lon = np.min(df['XLONG'].values)-0.25
    max_lon = np.max(df['XLONG'].values)+0.25
    for count,icomp in enumerate(compounds):
        output = f'{edgar_folder}/trimmed_edgar_HTAPv3_{edgar_year}_{icomp}.nc'
        if not os.path.exists(output):
            ds = xr.open_dataset(input)
            ds = ds.where((ds['lat'] >= min_lat) & (ds['lat'] <= max_lat) &
                     (ds['lon'] >= min_lon) & (ds['lon'] <= max_lon), drop=True)
            ds.to_netcdf(output)
            if clear_raw: os.remove(input)
        if (count==0)&(~os.path.exists(f'{edgar_folder}/area.nc')): 
            print(f'{edgar_folder}/area.nc')
            cdo.gridarea(input=output,output=f'{edgar_folder}/area.nc')
            print(f'Done')
            
var_dicts = {'ISO':
    {'description':'Isoprene EMISSIONS (Anth. for RADM/RACM, Anth+Bio for CBMZ)',
     'units':'mol km^-2 hr^-1'},
    'SO2':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'NO':
    {'description':'EMISSIONS',
    'units':'mol km^-2 hr^-1'},
    'NO2':
    {'description':'EMISSIONS NO2',
     'units':'mol km^-2 hr^-1'},
    'CO':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'ETH':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'HC3':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'HC5':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'HC8':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'XYL':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'OL2':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'OLT':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'OLI':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'TOL':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'CSL':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'HCHO':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'ALD':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'KET':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'ORA2':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'NH3':
    {'description':'EMISSIONS',
     'units':'mol km^-2 hr^-1'},
    'PM_25':
    {'description':'EMISSIONS',
     'units':'ug/m3 m/s'},
    'PM_10':
    {'description':'EMISSIONS',
     'units':'ug/m3 m/s'},
    'OC':
    {'description':'EMISSIONS OC AER',
     'units':'ug/m3 m/s'},
    'sulf':
    {'description':'EMISSIONS SO4',
     'units':'mol km^-2 hr^-1'},
    'BC':
    {'description':'EMISSIONS BC AER',
     'units':'ug/m3 m/s'}}

no_of_days_noleap = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
no_of_days_leap = {1:31,2:29,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

# The following corresponds to molar weights in g/mol units.
weights = {'O3':48.,'H2O2':34.,'NO':30.,'NO2':46.,'NO3':62.,'N2O5':108.,'HONO':47.,'HNO3':63.,'HNO4':79.,'SO2':64.,
           'SULF':98.,'CO':28.,'CO2':44.,'N2':28.,'O2':32.,'H2O':18.,'H2':2.,'O3P':16.,'O1D':16.,'HO':17.,'HO2':33.,
           'CH4':16.,'ETH':30.,'HC3':44.,'HC5':72.,'HC8':114.,'ETE':28.,'OLT':42.,'OLI':68.,'DIEN':54.,'ISO':68.,'API':136.,
           'LIM':136.,'TOL':92.,'XYL':106.,'CSL':108.,'HCHO':30.,'ALD':44.,'KET':72.,'GLY':58.,'MGLY':72.,
           'DCB':87.,'MACR':70.,'UDD':119.,'HKET':74.,'ONIT':119.,'PAN':121.,'TPAN':147.,'OP1':48.,'OP2':62.,
           'PAA':76.,'ORA1':46.,'ORA2':60.,'MO2':47.,'ETHP':61.,'HC3P':75.,'HC5P':103.,'HC8P':145.,'ETEP':77.,
           'OLTP':91.,'OLIP':117.,'ISOP':117.,'APIP':185.,'LIMP':185.,'PHO':107.,'ADDT':109.,'ADDX':123.,'ADDC':125.,
           'TOLP':141.,'XYLP':155.,'CSLP':157.,'ACO3':75.,'TCO3':115.,'KETP':103.,'OLNN':136.,'OLND':136.,'XO2':44.,
           'NH3':18.02,'PM25':00.,'PM10':00.,'BC':12.,'OC':12.,'DMS':62.,'ASH':-99.,'PM25':00.,'PM10':00.}

download_raw_emissions()

wrfinput_files = np.sort(glob(f'{wrfinput_folder}wrfinput*'))
wrfinpd01_file = [ifil for ifil in wrfinput_files if 'd01' in ifil][0]
df = xr.open_dataset(wrfinpd01_file)
trim_to_wrfinput(df,clear_raw=True)

df_speciation = pd.read_excel(reference_excel_htap,sheet_name='Regional NMVOC spec',skiprows=2,header=0)
df_geia_id = pd.read_excel(reference_excel_htap,sheet_name='NMVOC species',skiprows=2,header=0)
df_regions_edgar = pd.read_excel(reference_excel_htap,sheet_name='mapping country to regions',skiprows=1,header=0)
region = df_regions_edgar.loc[df_regions_edgar['Country name']==country,'Region definition'].values[0]
avail_edgar = glob(f'{edgar_folder}/trimmed*{edgar_year}*.nc') #File list of downloaded Edgar files
alloc_racm = pd.read_csv(alloc_racm_reference,skiprows=2,header=0,index_col='category')

avail_edgar_vars = [ivar.split('_')[-1].replace('.nc','') for ivar in avail_edgar]

wrfchemi_variables = [ivar for [ivar,value] in var_dicts.items()]
print("wrfchemi variables:",wrfchemi_variables)
bio_vars = ['ISO','OLI']
print("biogenic variables skilled:",bio_vars)
direct_vars = ['SO2','CO','NH3','PM_25','PM_10','OC','BC']
print("EDGAR vars loaded directly:",direct_vars)
nox_vars = ['NO','NO2']
nox_proportion = {'NO':0.7,'NO2':0.3}
print("NOX-related variables:",nox_vars)
nodata_var = ['OL2','sulf','CSL']
print("Empty variables:",nodata_var)
nmvoc_vars = list(set(wrfchemi_variables)-set(bio_vars+direct_vars+nodata_var+nox_vars))
print("NMVOC-relates variables:",nmvoc_vars)

wrfinpd = wrfinput_files[0]
print(f"Constructing {wrfinpd.replace('wrfinput','wrfchemi')}")
df = xr.open_dataset(wrfinpd)

edgar_chem_dicts = {'BC':'BC','CO':'CO','NH3':'NH3','OC':'OC','PM10':'PM_10','PM2.5':'PM_25','SO2':'SO2'}

df_out = df[['Times']]
df_out.attrs = df.attrs
area = xr.open_dataset(f'{edgar_folder}/area.nc')['cell_area']
for var in edgar_chem_dicts:
    df_edgar = xr.open_dataset(glob(f'{edgar_folder}/*{var}.nc')[0])
    edgar_htap = [ivar for ivar in df_edgar.variables if 'HTAP' in ivar]
    values = df_edgar[edgar_htap[0]]
    for edg in edgar_htap[1:]:
        values += df_edgar[edg]
    units = var_dicts[edgar_chem_dicts[var]]['units']
    ndays = no_of_days_noleap[month]
    if units == 'mol km^-2 hr^-1':
        values = values*1e6*1e6/weights[var]/ndays/24.0/area
    elif units == 'ug/m3 m/s':
        values = values*1e12/ndays/24.0/3600.0/area
    df_out[edgar_chem_dicts[var]] = values[month-1,:,:] ## The month is specified above.
    df_out[edgar_chem_dicts[var]].attrs = {}
    df_out[edgar_chem_dicts[var]].attrs['FieldType'] = 104
    df_out[edgar_chem_dicts[var]].attrs['MemoryOrder'] = "XYZ"
    df_out[edgar_chem_dicts[var]].attrs['description'] = var_dicts[edgar_chem_dicts[var]]['description']
    df_out[edgar_chem_dicts[var]].attrs['units'] = var_dicts[edgar_chem_dicts[var]]['units']
    df_out[edgar_chem_dicts[var]].attrs['stagger'] = ""
    df_out[edgar_chem_dicts[var]].attrs['coordinates'] = "XLONG XLAT XTIME"
    df_out = df_out.rename_vars({edgar_chem_dicts[var]:f'E_{edgar_chem_dicts[var]}'})

for var in nodata_var:
    df_out[var] = xr.DataArray(np.zeros_like(values[month-1,:,:]),dims=['lat','lon'])
    df_out[var].attrs['FieldType'] = 104
    df_out[var].attrs['MemoryOrder'] = "XYZ"
    df_out[var].attrs['description'] = var_dicts[var]['description']
    df_out[var].attrs['units'] = var_dicts[var]['units']
    df_out[var].attrs['stagger'] = ""
    df_out[var].attrs['coordinates'] = "XLONG XLAT XTIME"
    df_out = df_out.rename_vars({var:f'E_{var}'})   

for var in nox_vars:
    df_edgar = xr.open_dataset(glob(f'{edgar_folder}/*NOx.nc')[0])
    edgar_htap = [ivar for ivar in df_edgar.variables if 'HTAP' in ivar]
    values = df_edgar[edgar_htap[0]]
    for edg in edgar_htap[1:]:
        values += df_edgar[edg]
    units = var_dicts[var]['units']
    ndays = no_of_days_noleap[month]
    if units == 'mol km^-2 hr^-1':
        values = values*1e6*1e6/weights[var]/ndays/24.0/area
    elif units == 'ug/m3 m/s':
        values = values*1e12/ndays/24.0/3600.0/area
    df_out[var] = nox_proportion[var]*values[month-1,:,:] ## The month is specified above.
    df_out[var].attrs = {}
    df_out[var].attrs['FieldType'] = 104
    df_out[var].attrs['MemoryOrder'] = "XYZ"
    df_out[var].attrs['description'] = var_dicts[var]['description']
    df_out[var].attrs['units'] = var_dicts[var]['units']
    df_out[var].attrs['stagger'] = ""
    df_out[var].attrs['coordinates'] = "XLONG XLAT XTIME"
    df_out = df_out.rename_vars({var:f'E_{var}'})

#Some adjustments done to HC5. Refer to program.
speciation_coefficients = {'ORA2':{'fact1':[0.56],'contributor':['voc24']},
                'KET':{'fact1':[1.0],'contributor':['voc23']},
                'TOL':{'fact1':[1.0,1.0],'contributor':['voc13','voc14']},
                'HC8':{'fact1':[0.57,1.0],'contributor':['voc6','voc19']},
                'HCHO':{'fact1':[1.0],'contributor':['voc21']},
                'HC3':{'fact1':[0.95,1.0,1.0,1.0,1.0,0.69],'contributor':['voc1','voc9','voc3','voc4','voc20','voc18']},
                'ALD':{'fact1':[1.0],'contributor':['voc22']},
                'ETH':{'fact1':[1.0],'contributor':['voc2']},
                'HC5':{'fact1':[1.07*0.05/1.37,1.0,0.43,0.31],'contributor':['voc1','voc5','voc6','voc18']},
                'XYL':{'fact1':[1.0,1.0,1.0],'contributor':['voc15','voc16','voc17']},
                'OLT':{'fact1':[1.04],'contributor':['voc8']}}

df_speciation['VARIABLE']  = df_speciation['SECTOR CODE']+'_'+df_speciation['SECTOR NAME']
df_speciation['VARIABLE'] = df_speciation['VARIABLE'].apply(lambda x: x.replace('.','_').replace(' ','_').replace('HTAP','HTAPv3')
                                                            .replace('Road_transport','Road_Transport')
                                                            .replace('Domestic_Shipping','Domestic_shipping')
                                                            .replace('Agricultural_Waste_Burning','Agricultural_waste_burning')
                                                            .replace('International_shipping','International_Shipping')
                                                            .replace('International_aviation','International_Aviation')
                                                            .replace('Agricultural_crops','Agriculture_crops'))

def estimate_specie(edgar,racm,speciation,factor,contributor,region,month):
    for counter,ivar in enumerate(list(edgar.keys())):
        combined_weights = 0.0
        for ifact,icon in enumerate(contributor):
            if "International" in ivar:
                weighting = df_speciation.loc[(df_speciation['VARIABLE']==ivar)&(df_speciation['REGION']=='World'),icon].values[0]
            else:
                weighting = df_speciation.loc[(df_speciation['VARIABLE']==ivar)&(df_speciation['REGION']==region),icon].values[0]
            #Multiplied by racm
            weighting = weighting*np.float32(racm.loc[racm['edgarspec']==icon,'aggregation'].values[0])
            #Multiple by mass aggregation
            weighting = weighting*factor[ifact]
            if ifact == 0:
                combined_weights = weighting
            else:
                combined_weights = combined_weights + weighting
        if counter == 0:
            datavar = edgar[ivar][month-1,:,:]*combined_weights
        else:
            datavar = edgar[ivar][month-1,:,:]*combined_weights+datavar
    return datavar

df_edgar = xr.open_dataset(glob(f'{edgar_folder}/*NMVOC.nc')[0])
edgar_vars = [ivar for ivar in df_edgar.variables if 'HTAP' in ivar]
for var in nmvoc_vars:
    fact1,contributor = speciation_coefficients[var]['fact1'],speciation_coefficients[var]['contributor']
    values = estimate_specie(df_edgar,alloc_racm,df_speciation,fact1,contributor,region,month)
    values = values.rename(var)
    units = var_dicts[var]['units']
    ndays = no_of_days_noleap[month]
    if units == 'mol km^-2 hr^-1':
        values = values*1e6*1e6/weights[var]/ndays/24.0/area
    elif units == 'ug/m3 m/s':
        values = values*1e12/ndays/24.0/3600.0/area
    df_out[var] = values
    df_out[var].attrs = {}
    df_out[var].attrs['FieldType'] = 104
    df_out[var].attrs['MemoryOrder'] = "XYZ"
    df_out[var].attrs['description'] = var_dicts[var]['description']
    df_out[var].attrs['units'] = var_dicts[var]['units']
    df_out[var].attrs['stagger'] = ""
    df_out[var].attrs['coordinates'] = "XLONG XLAT XTIME"
    df_out = df_out.rename_vars({var:f'E_{var}'})
df_out = df_out.drop(['time'])

def resample_to_final_conservative(df_out,wrfinput_folder):
    wrfinps = glob(f'{wrfinput_folder}wrfinput_d*')
    for iwrf in wrfinps:
        df = xr.open_dataset(iwrf)
        df['lon'] = df['XLONG'][0,:,:]
        df['lat'] = df['XLAT'][0,:,:]
        df = df.cf.add_bounds('lon')
        df = df.cf.add_bounds('lat')
        dimensions = {
            "lat": (('y','x'),df['XLAT'][0,:,:].values),
            "lon": (('y','x'),df['XLONG'][0,:,:].values),
            "lat_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lat_bounds'],bounds_dim='bounds').values),
            "lon_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lon_bounds'],bounds_dim='bounds').values)
        }
        df_target = xr.Dataset(coords=dimensions)
        regridder = xe.Regridder(df_out[['lat','lon']].rename({'lat':'latitude','lon':'longitude'}), df_target, 'conservative')#,ignore_degenerate=True)
        df_out_final = regridder(df_out.rename({'lat':'latitude','lon':'longitude'}))
        df_out_final['Times'] = df_out['Times']
        df_out_final.attrs = df_out.attrs
        df_out_final = df_out_final.expand_dims(dim='emissions_zdim') #Add vertial level dimension. Currently surface
        #Add vertial level dimension. Currently surface
        for ivar in df_out_final.keys():
            if 'E_' in ivar:
                df_out_final[ivar] = df_out_final[ivar].expand_dims(dim='Time') 
        df_out_final = df_out_final.rename({'y':'south_north','x':'west_east'}) #Rename to match wrfchemi_d*
        df_out_final.transpose('Time','emissions_zdim','south_north','west_east') #Reorder the dimensions
        for ivar in df_out_final.keys():
            if 'E_' in ivar:
                df_out_final[ivar].attrs = {}
                df_out_final[ivar].attrs['FieldType'] = 104
                df_out_final[ivar].attrs['MemoryOrder'] = "XYZ"
                df_out_final[ivar].attrs['description'] = df_out[ivar].attrs['description']
                df_out_final[ivar].attrs['units'] = df_out[ivar].attrs['units']
                df_out_final[ivar].attrs['stagger'] = ""
                df_out_final[ivar].attrs['coordinates'] = "XLONG XLAT XTIME"
        df_out_final.drop(['lat','lon']).to_netcdf(out_folder+os.path.basename(iwrf.replace('input_','chemi_')))

def resample_to_final(df_out,wrfinput_folder,verbose=True):
    wrfinps = glob(f'{wrfinput_folder}wrfinput_d*')
    for iwrf in wrfinps:
        df = xr.open_dataset(iwrf)
        df['lon'] = df['XLONG'][0,:,:]
        df['lat'] = df['XLAT'][0,:,:]
        df = df.cf.add_bounds('lon')
        df = df.cf.add_bounds('lat')
        dimensions = {
            "lat": (('y','x'),df['XLAT'][0,:,:].values),
            "lon": (('y','x'),df['XLONG'][0,:,:].values),
            "lat_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lat_bounds'],bounds_dim='bounds').values),
            "lon_b": (('y_b','x_b'),cf_xarray.bounds_to_vertices(df['lon_bounds'],bounds_dim='bounds').values)
        }
        df_target = xr.Dataset(coords=dimensions)
        raw_spacing = np.diff(df_out['lon']).mean()
        wrf_spacing = np.diff(df_target['lon']).mean()
        if (raw_spacing >= wrf_spacing):
            resampler = 'bilinear'
        else:
            resampler = 'conservative'
        if verbose: print(f"note: {iwrf} will be regridded by {resampler}.")
        regridder = xe.Regridder(df_out[['lat','lon']].rename({'lat':'latitude','lon':'longitude'}), df_target, resampler)#,ignore_degenerate=True)
        df_out_final = regridder(df_out.rename({'lat':'latitude','lon':'longitude'}))
        df_out_final['Times'] = df_out['Times']
        df_out_final.attrs = df_out.attrs
        df_out_final = df_out_final.expand_dims(dim='emissions_zdim') #Add vertial level dimension. Currently surface
        #Add vertial level dimension. Currently surface
        for ivar in df_out_final.keys():
            if 'E_' in ivar:
                df_out_final[ivar] = df_out_final[ivar].expand_dims(dim='Time') 
        df_out_final = df_out_final.rename({'y':'south_north','x':'west_east'}) #Rename to match wrfchemi_d*
        df_out_final.transpose('Time','emissions_zdim','south_north','west_east') #Reorder the dimensions
        for ivar in df_out_final.keys():
            if 'E_' in ivar:
                df_out_final[ivar].attrs = {}
                df_out_final[ivar].attrs['FieldType'] = 104
                df_out_final[ivar].attrs['MemoryOrder'] = "XYZ"
                df_out_final[ivar].attrs['description'] = df_out[ivar].attrs['description']
                df_out_final[ivar].attrs['units'] = df_out[ivar].attrs['units']
                df_out_final[ivar].attrs['stagger'] = ""
                df_out_final[ivar].attrs['coordinates'] = "XLONG XLAT XTIME"
        df_out_final.drop(['lat','lon']).to_netcdf(os.path.basename(iwrf.replace('input_','chemi_')))
resample_to_final(df_out,wrfinput_folder)
