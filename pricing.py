
"""
Created on Mon Aug  8 16:46:26 2022
@author: Arvin Jay
"""

import pandas as pd
import numpy as np
import math, re
import gspread, datetime
import os, sys

from decimal import Decimal
from fuzzywuzzy import fuzz, process

import streamlit as st
from st_aggrid import AgGrid,GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


from io import BytesIO
from google.oauth2 import service_account
#from gsheetsdb import connect


credentials = service_account.Credentials.from_service_account_info(
st.secrets["lica_service_account"],
scopes=[
    "https://www.googleapis.com/auth/spreadsheets",
],)

creds = st.secrets['lica_service_account']
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
output_path = os.path.abspath(os.path.dirname(__file__))
os.chdir(output_path) # current working directory

@st.cache_data
def implement_sale(df, sale_tag, promo, srp):
    df.loc[df[sale_tag]==0,promo] =df.loc[df[sale_tag]==0,srp]
    return df

def to_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def consider_GP(data,GP):
    return ceil_5(data/(1-float(GP)/100))

def get_GP(qsupp,qsp):
  return round(100*(1- (float(qsupp)/float(qsp))),2)

def ceil_5(n):
    return math.ceil(n/5)*5

def update():
    st.cache_data.clear()
    del st.session_state['adjusted']
    del st.session_state['GP_15']
    del st.session_state['GP_20']
    del st.session_state['GP_20_']
    st.experimental_rerun()

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')
    
def highlight_promo(xa):
    df1 = pd.DataFrame('background-color: ', index=xa.index, columns=xa.columns)
    col_eval = ['GulongPH','GulongPH_slashed','b2b','marketplace']
    highlight_competitor = '#ffffb3'
    temp_list = list(col_tier)
    col_eval = col_eval+temp_list
    for column in col_eval:
        c = xa['supplier_max_price'] > xa[column]
        df1['supplier_max_price']= np.where(c, 'background-color: {}'.format('pink'), df1['supplier_max_price'])
        df1[column]= np.where(c, 'background-color: {}'.format('pink'), df1[column])
    if 'selection_max_price' in xa.columns.tolist():
        c = xa['selection_max_price']<xa['supplier_max_price']
        df1['selection_max_price'] = np.where(c, 'background-color: {}'.format('lightgreen'), df1['selection_max_price'])
    if 'GoGulong' in xa.columns.tolist():
        
        c = xa['GulongPH']>xa['GoGulong']
        df1['GulongPH'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GulongPH'])
        df1['GoGulong'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GoGulong'])
    if 'TireManila' in xa.columns.tolist():
        c = xa['GulongPH']>xa['TireManila']
        df1['TireManila'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['TireManila'])
        df1['GulongPH'] = np.where(c, 'background-color: {}'.format(highlight_competitor), df1['GulongPH'])
    return df1

def highlight_smallercompetitor(xa):
    df1 = pd.DataFrame('background-color: ', index=xa.index, columns=xa.columns)
    col_eval = ['GoGulong','TireManila','PartsPro']
    for column in col_eval:
        if column in xa.columns:
            c = xa['GulongPH'] > xa[column]
            df1['GulongPH']= np.where(c, 'background-color: {}'.format('pink'), df1['GulongPH'])
            df1[column]= np.where(c, 'background-color: {}'.format('pink'), df1[column])
    return df1

def highlight_others(x):#cols = ['GP','Tier 1','Tier 3', etc]
    df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
    c1 = x['GulongPH'] > x['GulongPH_slashed']
    df1['GulongPH']= np.where(c1, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH'])
    df1['GulongPH_slashed']= np.where(c1, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH_slashed'])
    c2 = x['marketplace']<x['GulongPH']
    df1['GulongPH']= np.where(c2, 'color:{};font-weight:{}'.format('red','bold'), df1['GulongPH'])
    df1['marketplace']= np.where(c2, 'color:{};font-weight:{}'.format('red','bold'), df1['marketplace'])
    return df1
  
def df_numeric(df_temp):
    for cols in df_temp.columns.to_list():
        df_temp.loc[:,cols] = df_temp.loc[:,cols].apply(pd.to_numeric)
    return df_temp

def filter_data_captured(df_test, tier):
    df_compet =pd.DataFrame()
    if 'GoGulong' in df_test.columns.tolist():
        df_temp = df_test.loc[df_test['GulongPH']>df_test['GoGulong']]
        df_compet = pd.concat([df_compet,df_temp])
    if 'TireManila' in df_test.columns.tolist():
        df_temp = df_test.loc[df_test['GulongPH']>df_test['TireManila']]
        df_compet = pd.concat([df_compet,df_temp])
    df_A = df_test.loc[df_test['supplier_max_price']> df_test[['GulongPH','b2b','marketplace']].min(axis=1)]
    df_B = df_test.loc[df_test['GulongPH']> df_test[['GulongPH_slashed','marketplace']].min(axis=1)]
    df_C = pd.DataFrame()
    for col in tier:
        df_E = df_test.loc[df_test['supplier_max_price']> df_test[col]]
        df_C = pd.concat([df_C,df_E],axis=0)
    df_show = pd.concat([df_A,df_compet,df_B,df_C],axis=0)
    df_show = df_show.drop_duplicates()
    return df_show

def build_grid(df_show):
    gb = GridOptionsBuilder.from_dataframe(df_show)
    gb.configure_default_column(enablePivot=False, enableValue=False, enableRowGroup=False, editable = True)
    gb.configure_column('model', headerCheckboxSelection = True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()  
    gridOptions = gb.build()
    return gridOptions

def promotize(value, GP_promo):
    return consider_GP(value*4/3,float(GP_promo))

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def adjust_wrt_gogulong(df_comp,GP_15=15,GP_20=5,GP_20_=1, b2b=25,affiliate=27,mp=25):
  df_comp_ = pd.DataFrame()
  df_comp_ = df_comp.loc[df_comp['GulongPH'] > df_comp['GoGulong']].copy()
  if len(df_comp_)==0:
      return df_comp_
  df_comp_['GoGulong_GP'] = df_comp_.apply(lambda x: get_GP(x['supplier_max_price'],x['GoGulong']),axis=1)

  #when gogulong GP <= 15%, set gulong GP to 15%
  df_comp_.loc[df_comp_['GoGulong_GP']<15,'GulongPH'] = df_comp_.loc[df_comp_['GoGulong_GP']<15,'supplier_max_price'].apply(lambda x: consider_GP(x,GP_15))

  #when gogulong GP <= 20%, set gulong price = gogulong price (floor_5)
  df_comp_.loc[(df_comp_['GoGulong_GP']<20) & (df_comp_['GoGulong_GP']>=15),'GulongPH'] = df_comp_.loc[(df_comp_['GoGulong_GP']<20) & (df_comp_['GoGulong_GP']>=15),'GoGulong'].apply(lambda x: ceil_5(x-GP_20))

  #when gogulong GP >20%, set gulong price GP =  gogulong price GP - 1
  df_comp_.loc[df_comp_['GoGulong_GP']>=20,'GulongPH'] = df_comp_.loc[df_comp_['GoGulong_GP']>=20,:].apply(lambda x: consider_GP(x['supplier_max_price'], x['GoGulong_GP']-GP_20_),axis=1) #math.ceil(x['GoGulong_GP']-GP_20_))
  
  df_comp_['GulongPH_GP'] = df_comp_.apply(lambda x: get_GP(x['supplier_max_price'],x['GulongPH']),axis=1)
  df_comp_.loc[:,'GulongPH_slashed'] = df_comp_.loc[:,'supplier_max_price'].apply(lambda x: consider_GP(x,30))
  df_comp_.loc[:,'b2b'] = df_comp_.loc[:,'supplier_max_price'].apply(lambda x: consider_GP(x,b2b))
  df_comp_.loc[:,'affiliate'] = df_comp_.loc[:,'supplier_max_price'].apply(lambda x:  consider_GP(x,affiliate))
  df_comp_.loc[:,'marketplace'] = df_comp_.loc[:,'supplier_max_price'].apply(lambda x:  consider_GP(x,mp))
  return df_comp_

def import_makes():
    '''
    Import list of makes
    '''
    with open("gulong_makes.txt") as makes_file:
        makes = makes_file.readlines()
        
    makes = [re.sub('\n', '', m).strip() for m in makes]
    return makes

makes_list = import_makes()

def clean_make(x, makes, model = None):
    '''
    Helper function to cleanup gulong makes
    
    Parameters:
        - x: string; value of gulong make, can be NaN
        - makes: list; reference list of gulong makes from gulong_makes.txt
    
    Output:
        cleaned string value of gulong make
    '''
    if pd.notna(x):
        # baseline string correction if not NaN
        x = x.strip().upper()
        # check if partial match exists
        if any((match := m) for m in makes if fuzz.partial_ratio(m, x) >= 95):
            return match
        # if model is provided, check model if make is included
        elif model is not None and any((match := m) for m in makes if fuzz.partial_ratio(m, model.upper().strip()) >= 95):
            return match
        # check for best match given threshold
        elif process.extractOne(x, makes)[1] >= 75:
            return process.extractOne(x, makes)[0]
        else:
            return x
    else:
        # if input is NaN
        return np.NaN

def remove_trailing_zero(num):
    '''
    Removes unnecessary zeros from decimals

    Parameters
    ----------
    num : Decimal(number)
        number applied with Decimal function (see import decimal from Decimal)

    Returns
    -------
    number: Decimal
        Fixed number in Decimal form

    '''
    return num.to_integral() if num == num.to_integral() else num.normalize()

def clean_width(w):
    '''
    Clean width values
    
    Parameters
    ----------
    d: string
        width values in string format
        
    Returns:
    --------
    d: string
        cleaned diameter values
    
    DOCTESTS:
    >>> clean_width('175')
    '175'
    >>> clean_width('6.50')
    '6.5'
    >>> clean_width('27X')
    '27'
    >>> clean_width('LT35X')
    'LT35'
    >>> clean_width('8.25')
    '8.25'
    >>> clean_width('P265.5')
    'P265.5'
    >>> clean_width(np.NaN)
    nan
    
    '''
    if pd.notna(w):
        w = str(w).strip().upper()
        # detects if input has expected format
        prefix_num = re.search('[A-Z]*[0-9]+.?[0-9]*', w)
        if prefix_num is not None:
            num_str = ''.join(re.findall('[0-9]|\.', prefix_num[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            prefix = w.split(num_str)[0]
            return prefix + num
        else:
            return np.NaN
    else:
        return np.NaN
    
def clean_diameter(d):
    '''
    Fix diameter values
    
    Parameters
    ----------
    d: string
        diameter values in string format
        
    Returns:
    --------
    d: string
        fixed diameter values
    
    DOCTESTS:
    >>> clean_diameter('R17LT')
    'R17LT'
    >>> clean_diameter('R22.50')
    'R22.5'
    >>> clean_diameter('15')
    'R15'
    >>> clean_diameter(np.NaN)
    nan
    
    '''
    if pd.notna(d):
        d = str(d).strip().upper()
        num_suffix = re.search('[0-9]+.?[0-9]*[A-Z]*', d)
        if num_suffix is not None:
            num_str = ''.join(re.findall('([0-9]|\.)', num_suffix[0]))
            num = str(remove_trailing_zero(Decimal(num_str)))
            suffix = num_suffix[0].split(num_str)[-1]
            return f'R{num}{suffix}'
    else:
        return np.NaN

def clean_aspect_ratio(ar, model = None):
    
    '''
    Clean raw aspect ratio input
    
    Parameters
    ----------
    ar: float or string
        input raw aspect ratio data
    model: string; optional
        input model string value of product
        
    Returns
    -------
    ar: string
        fixed aspect ratio data in string format for combine_specs
    
    DOCTESTS:
    >>> clean_aspect_ratio('/')
    'R'
    >>> clean_aspect_ratio('.5')
    '9.5'
    >>> clean_aspect_ratio('14.50')
    '14.5'
    >>> clean_aspect_ratio(np.NaN)
    'R'
    
    '''
    error_ar = {'.5' : '9.5',
                '0.': '10.5',
                '2.': '12.5',
                '3.': '13.5',
                '5.': '15.5',
                '70.5': '10.5'}
    
    if pd.notna(ar):
        # aspect ratio is faulty
        if str(ar) in ['0', 'R1', '/', 'R']:
            return 'R'
        # incorrect parsing of decimal aspect ratios
        elif str(ar) in error_ar.keys():
            return error_ar[str(ar)]
        # numeric/integer aspect ratio
        elif str(float(ar)).isnumeric():
            return str(ar)
        # decimal aspect ratio with trailing 0
        else:
            return str(remove_trailing_zero(Decimal(str(ar))))
    else:
        return 'R'

def combine_specs(w, ar, d, mode = 'SKU'):
    '''
    
    Parameters
    - w: string
        section_width
    - ar: string
        aspect_ratio
    - d: string
        diameter
    - mode: string; optional
        SKU or MATCH
    
    Returns
    - combined specs with format for SKU or matching
    
    >>> combine_specs('175', 'R', 'R15', mode = 'SKU')
    '175/R15'
    >>> combine_specs('175', '65', 'R15', mode = 'SKU')
    '175/65/R15'
    >>> combine_specs('33', '12.5', 'R15', mode = 'SKU')
    '33X12.5/R15'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'SKU')
    'LT175/65/R15C'
    >>> combine_specs('LT175', '65', 'R15C', mode = 'MATCH')
    '175/65/15'
    
    '''
    if mode == 'SKU':
        d = d if 'R' in d else 'R' + d 
        if ar != 'R':
            if '.' in ar:
                return w + 'X' + ar + '/' + d
            else:
                return '/'.join([w, ar, d])
        else:
            return w + '/' + d
            
    elif mode == 'MATCH':
        w = ''.join(re.findall('[0-9]|\.', w))
        ar = ''.join(re.findall('[0-9]|\.|R', ar))
        d = ''.join(re.findall('[0-9]|\.', d))
        return '/'.join([w, ar, d])

    else:
        combine_specs(w, ar, d, mode = 'SKU')

def clean_speed_rating(sp):
    '''
    Clean speed rating of gulong products
    
    DOCTESTS:
    >>> clean_speed_rating('W XL')
    'W'
    >>> clean_speed_rating('0')
    'B'
    >>> clean_speed_rating('118Q')
    'Q'
    >>> clean_speed_rating('T/H')
    'T'
    >>> clean_speed_rating('-')
    nan
    
    '''
    # SAILUN 205/75/R16C COMMERCIO VX1 10PR - 113/111R
    # SAILUN 205/75/R16C COMMERCIO VX1 8PR - 110/108R
    # SAILUN 235/65/R16C COMMERCIO VX1 8PR - 115/113R
    # SAILUN 33X/12.50/R20 TERRAMAX M/T 10PR - 114Q
    # SAILUN 35X/12.50/R20 TERRAMAX M/T 10PR - 121Q
    # SAILUN 305/55/R20 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 35X/12.50/R18 TERRAMAX M/T 10PR - None
    # SAILUN 33X/12.50/R18 TERRAMAX M/T 10PR - 118Q
    # SAILUN 35X/12.50/R17 TERRAMAX M/T 10PR - 121Q
    # SAILUN 33X/12.50/R17 TERRAMAX M/T 8PR - 114Q
    # SAILUN 285/70/R17 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 265/70/R17 TERRAMAX M/T 10PR - 121/118Q
    # SAILUN 265/75/R16 TERRAMAX M/T 10PR - 116S
    # SAILUN 245/75/R16 TERRAMAX M/T 10PR - 111S
    # SAILUN 35X/12.50/R15 TERRAMAX M/T 6PR - 113Q
    # SAILUN 33X/12.50/R15 TERRAMAX M/T 6PR - 108Q
    # SAILUN 31X/10.50/R15 TERRAMAX M/T 6PR - 109S
    # SAILUN 30X/9.50/R15 TERRAMAX M/T 6PR - 104Q
    # SAILUN 235/75/R15 TERRAMAX M/T 6PR - 104/101Q
    # SAILUN 265/70/R17 TERRAMAX A/T 10PR - 121/118S
    
    # not NaN
    if pd.notna(sp):
        # baseline correct
        sp = sp.strip().upper()
        # detect if numerals are present 
        num = re.search('[0-9]{2,3}', sp)
        
        if num is None:
            pass
        else:
            # remove if found
            sp = sp.split(num[0])[-1].strip()
            
        if 'XL' in sp:
            return sp.split('XL')[0].strip()
        elif '/' in sp:
            return sp.split('/')[0].strip()
        elif sp == '0':
            return 'B'
        elif sp == '-':
            return np.NaN
        else:
            return sp
    else:
        return np.NaN

def clean_load_rating(load):
    
    # SAILUN 225/60/R17 TERRAMAX CVR 199H - 99H instead of 199H
    # SAILUN 225/60/R17 TERRAMAX CVR 199H - 91W instead of 914W
    
    return 0

def combine_sku(row):
    df = df = pd.DataFrame(columns = ['make', 
                                      'section_width', 
                                      'aspect_ratio', 
                                      'rim_size', 
                                      'pattern', 
                                      'load_rating', 
                                      'speed_rating'])
    df = df.append(pd.Series({'make': 'ARIVO', 
                              'section_width': '195', 
                              'aspect_ratio': 'R',
                              'rim_size': 'R15',
                              'pattern': 'TRANSITO ARZ 6-X',
                              'load_rating': '106/104',
                              'speed_rating': 'Q'}), ignore_index = True)
    
    '''
    DOCTESTS:
            
    >>> combine_sku(df.loc[0])
    'ARIVO 195/R15 TRANSITO ARZ 6-X 106/104Q'
    
    '''
    specs_cols = ['section_width', 'aspect_ratio', 'rim_size']
    specs = combine_specs(row[specs_cols[0]], 
                          row[specs_cols[1]], 
                          row[specs_cols[2]],
                          mode = 'SKU')
    
    if pd.notna(row['pattern']):
        SKU = ' '.join([row['make'], specs, row['pattern']])
    else:
        SKU = ' '.join([row['make'], specs])
    
    if pd.notna(row['load_rating']) and pd.notna(row['speed_rating']):
        SKU  = SKU + ' ' +  row['load_rating'] + row['speed_rating']
    else:
        pass
    return SKU
    
@st.cache_data
def acquire_data():
    # http://app.redash.licagroup.ph/queries/131
    url1 =  "http://app.redash.licagroup.ph/api/queries/131/results.csv?api_key=BdUhcTVmwDEqP5aYKpSolS5ApT2lig4hpdDqIPJq"

    df_data = pd.read_csv(url1, parse_dates = ['supplier_price_date_updated','product_price_date_updated'])
    #df_data.loc[df_data['sale_tag']==0,'promo'] =df_data.loc[df_data['sale_tag']==0,'srp']
    df_data = df_data[['make','model', 'section_width', 'aspect_ratio', 'rim_size' ,'pattern', 'load_rating','speed_rating','stock','name','cost','srp', 'promo', 'mp_price','b2b_price' , 'supplier_price_date_updated','product_price_date_updated','supplier_id','sale_tag']]
    df_data.columns = ['make','model', 'section_width', 'aspect_ratio', 'rim_size','pattern', 'load_rating','speed_rating','stock','supplier','supplier_price','GulongPH_slashed','GulongPH','marketplace','b2b','supplier_updated','gulong_updated','supplier_id','sale_tag']
    
    # cleaning
    df_data.loc[:, 'make'] = df_data.apply(lambda x: clean_make(x['make'], makes_list, model = x['model']), axis=1)
    df_data.loc[:, 'section_width'] = df_data.apply(lambda x: clean_width(x['section_width']), axis=1)
    df_data.loc[:, 'aspect_ratio'] = df_data.apply(lambda x: clean_aspect_ratio(x['aspect_ratio'], model = x['model']), axis=1)
    df_data.loc[:, 'rim_size'] = df_data.apply(lambda x: clean_diameter(x['rim_size']), axis=1)
    df_data.loc[:, 'speed_rating'] = df_data.apply(lambda x: clean_speed_rating(x['speed_rating']), axis=1)
    df_data.loc[:, 'model_'] = df_data.loc[:, 'model']
    df_data.loc[:, 'model'] = df_data.apply(lambda x: combine_sku(x), axis=1)
    
    df_supplier = df_data[['model','supplier','supplier_price','supplier_updated']].copy().sort_values(by='supplier_updated',ascending=False)
    df_supplier = df_supplier.drop_duplicates(subset=['model','supplier'],keep='first')
    df_supplier = df_supplier.groupby(['model','supplier'], group_keys=False).agg(price = ('supplier_price',lambda x: x))
    df_supplier = df_supplier.unstack('supplier').reset_index().set_index(['model'])
    df_supplier.columns =[i[1] for i in df_supplier.columns] 
    df_supplier['supplier_max_price'] = df_supplier.fillna(0).max(axis=1)
    df_supplier = df_supplier.reset_index()
    
    df_gulong = df_data[['make','model', 'section_width', 'aspect_ratio', 'rim_size','pattern', 'load_rating','speed_rating','GulongPH_slashed','GulongPH','b2b','marketplace','gulong_updated','stock','supplier_id','sale_tag']].copy().sort_values(by='gulong_updated',ascending=False)
    df_gulong = df_gulong.drop_duplicates(subset='model',keep='first').drop('gulong_updated',axis = 1)

    # import scraped competitor data    
    gsheet_key = "12jCVn8EQyxXC3UuQyiRjeKsA88YsFUuVUD3_5PILA2c"
    gc = gspread.service_account_from_dict(creds)
    sh = gc.open_by_key(gsheet_key)
    sheet_list = []
    worksheet_list = sh.worksheets()
    for item in range(len(worksheet_list)):
      if 'Copy' not in worksheet_list[item].title:
          sheet_list.append(worksheet_list[item].title)
    get_sheet = max(sheet_list)
    worksheet = sh.worksheet(get_sheet)
    
    # sheet_url = st.secrets["gsheets_urlB"]
    # rows = run_query(f'SELECT * FROM "{sheet_url}"')
    # df_competitor = pd.DataFrame(rows)
    
    df_competitor = pd.DataFrame(worksheet.get_all_records())
    df_competitor = df_competitor[['sku_name','price_gogulong','price_tiremanila','price_partspro']]
    df_competitor.columns = ['model', 'GoGulong','TireManila','PartsPro']
    df_competitor = df_competitor.replace('',np.nan)
    df_competitor['GoGulong_slashed'] = df_competitor['GoGulong'].apply(lambda x: float(x)/0.8)
    
    df_temp = df_gulong.merge(df_supplier, on = 'model', how='outer').merge(df_competitor, on='model', how='left').sort_values(by='model')
    df_temp = df_temp.dropna(subset = 'supplier_max_price')
    df_temp['dimensions'] = df_temp.apply(lambda x: '/'.join(x[['section_width','aspect_ratio','rim_size']].astype(str)),axis=1)
    df_temp['GoGulong_GP'] = df_temp.loc[:,['supplier_max_price','GoGulong']].apply(lambda x: round(get_GP(x['supplier_max_price'],x['GoGulong']),2),axis=1)
    df_temp = df_temp.loc[df_temp['GulongPH'] !=0]
    df_temp['GulongPH_GP'] = df_temp.loc[:,['supplier_max_price','GulongPH']].apply(lambda x: round(get_GP(x['supplier_max_price'],x['GulongPH']),2),axis=1)
    cols_option = ['GoGulong','GoGulong_slashed','TireManila','PartsPro','GoGulong_GP','GulongPH_GP'] + list(df_supplier.columns)
    df_temp['3+1_promo_per_tire_GP25'] = df_temp['supplier_max_price'].apply(lambda x: promotize(x,25))
    df_temp = df_temp.drop_duplicates(subset='model',keep='first')
    if 'model' in cols_option:
        cols_option.remove('model')
    if 'supplier_max_price' in cols_option:
        cols_option.remove('supplier_max_price')
    #my_bar.progress(100)
    #bar_container.empty()
    if 'updated_at' not in st.session_state:
        st.session_state['updated_at'] = datetime.datetime.today().date()
    return df_temp, cols_option, df_competitor, get_sheet

if 'GP_15' not in st.session_state:
    st.session_state['GP_15'] = 15
if 'GP_20' not in st.session_state:
    st.session_state['GP_20'] = 5
if 'GP_20_' not in st.session_state:
    st.session_state['GP_20_']= 3
if 'd_b2b' not in st.session_state:
    st.session_state['d_b2b'] = 25
if 'd_affiliate' not in st.session_state:
    st.session_state['d_affiliate'] = 27
if 'd_marketplace' not in st.session_state:
    st.session_state['d_marketplace'] = 25
    
if 'adjusted' not in st.session_state:
    st.session_state['adjusted'] = False

qc_expander = st.sidebar.expander("Quick Calculator")
with qc_expander:
    find_value = st.radio("Find:", ('Selling Price','Supplier Price', 'GP(%)'))
    q1,q2 = st.columns([1,1])
    if find_value =='Selling Price':     
        with q1:
            qsp = st.text_input('Input Supplier Price:', value="1000.00")
        with q2:
            qgp = st.text_input('Input GP: (%)', value="30.00")
            if to_float(qgp) and to_float(qsp):
                value = consider_GP(float(qsp),float(qgp))
            else:
                value = "Input Error"
    if find_value =='Supplier Price':       
        with q1:
            qsp = st.text_input('Input Selling Price:', value="1000.00")
        with q2:
            qgp = st.text_input('Input GP (%):', value="30.00")
            if to_float(qgp) and to_float(qsp):
                value = round(float(qsp)*(1-float(qgp)/100),)
            else:
                value = "Input Error"
    if find_value == 'GP(%)':
        with q1:
            qsp = st.text_input('Input Selling Price:', value="1000.00")
        with q2:
            qsupp = st.text_input('Input Supplier Price:', value="1500.00")
            if (to_float(qsupp) and to_float(qsp)):
                if float(qsp)==0:
                    value = "Input Error"
                else:
                    value = get_GP(qsupp,qsp)
            else:
                value = "Input Error"
    st.metric(find_value, value)      



t_name= st.sidebar.expander("Rename Tiers:")
with t_name:
    t1_name = st.text_input('Tier 1 name:', 'Website Slashed Price Test')
    t2_name = st.text_input('Tier 2 name:', 'Website Prices Test')
    t3_name = st.text_input('Tier 3 name:', 'B2B Test')
    t4_name = st.text_input('Tier 4 name:', 'Marketplace Test')
    t5_name = st.text_input('Tier 5 name:', 'Affiliates Test')

df_final, cols_option,df_competitor, last_update = acquire_data()

CS1a,CS2a = st.sidebar.columns([2,3])

if 'updated_at' not in st.session_state:
    update()

with CS1a:
    if st.button('Update Data',help='Acquires resutls of query from backend and resets program'):
        update()
with CS2a:
    st.caption('GulongPH data last updated on'+str(st.session_state['updated_at']))
    st.caption('GoGulong/TireManila data last updated on '+str(last_update))

if (st.session_state['updated_at'] !=datetime.datetime.today().date()):
    update()

st.sidebar.markdown("""---""")
CS1,CS2 = st.sidebar.columns([2,3])
with CS1:
    is_adjusted = st.checkbox('Auto-adjust')
with CS2:
    st.caption('Automatically adjusts data based on GoGulong values')
if is_adjusted:
    st.session_state['adjusted'] = True
else:
    st.session_state['adjusted'] = False

edit_mode = st.sidebar.selectbox('Mode', options = ('Automated','Manual'),index = 1)

if st.session_state['adjusted']:
    df_final_ = implement_sale(df_final, 'sale_tag', 'GulongPH', 'GulongPH_slashed').drop(columns= 'sale_tag')
    df_final_= df_final_.set_index('model')
    df_temp_adjust = adjust_wrt_gogulong(df_final_,st.session_state['GP_15'],
                                                    st.session_state['GP_20'],
                                                    st.session_state['GP_20_'],
                                                    st.session_state['d_b2b'],
                                                    st.session_state['d_affiliate'],
                                                    st.session_state['d_marketplace'],)
    df_final_.update(df_temp_adjust[['GulongPH','GulongPH_slashed','b2b','affiliate','marketplace']], overwrite = True)
    df_final_= df_final_.reset_index()
else:
    df_final_ = df_final.copy()
    
if edit_mode == 'Manual':
    st.header("Data Review")
    
    with st.expander('Include/remove columns in list:'):
        beta_multiselect = st.container()
        check_all = st.checkbox('Select all', value=False)
        if check_all:
            selected_supplier_ = beta_multiselect.multiselect('Included columns in table:',
                                           options = cols_option,
                                           default = list(cols_option))
        else:
            selected_supplier_ = beta_multiselect.multiselect('Included columns in table:',
                                           options = cols_option)
            
    cols = ['model','make','dimensions','supplier_max_price','3+1_promo_per_tire_GP25','GulongPH','GulongPH_slashed','b2b','marketplace']
    
    df_show =df_final_[cols].merge(df_final[['model','GulongPH']], how = 'left',left_on = 'model',right_on = 'model', suffixes=('','_backend'))
    check_adjusted = st.sidebar.checkbox('Show adjusted prices only', value = False)
    if check_adjusted:
        df_show = df_show.loc[df_show['GulongPH'] != df_show['GulongPH_backend']]
    if len(selected_supplier_) >0:
        cols.extend(selected_supplier_)
        df_show =df_final_[cols].dropna(how = 'all', subset = selected_supplier_,axis=0).replace(np.nan,'')
    
    st.write("Select the SKUs that would be considered for the computations.",
             " Feel free to filter the _make_ and _model_ that would be shown. You may also select/deselect columns.")
    
    reload_data = False
    
    if st.sidebar.button('Reset changes'):
        reload_data = True
    st.sidebar.caption('Resets the edits done in the table.')
    
    gridOptions = build_grid(df_show)
    response = AgGrid(df_show,
        #theme = 'light',
        gridOptions=gridOptions,
        height = 300,
        #width = '100%',
        editable=True,
        allow_unsafe_jscode=True,
        reload_data=reload_data,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False)
    reload_data = False
    
    AG1, AG2 =st.columns([3,2])
    with AG1:
        st.write("Results: "+str(len(df_show))+" entries")
    with AG2:
        st.download_button(label="📥 Download this table.",
                            data=convert_df(pd.DataFrame.from_dict(response['data'])),
                            file_name='grid_table.csv',
                            mime='text/csv')
              
    st.markdown("""
                ---
                """)

    st.header("Price Comparison")   
    st.write("You may set the GP and the price comparison between models would be shown in a table.")
    
    ct1, ct2,ct3,ct4,ct5, cs3= st.columns([1,1,1,1,1,1])#,cS,cs1,cs2,cs3
    with ct1:
        test_t1 = st.checkbox(t1_name)
        t1_GP = st.text_input("GP (%):", value = "30", key='t1')
    with ct2:
        test_t2 = st.checkbox(t2_name)
        t2_GP = st.text_input("GP (%):", value = "27", key='t2')
    with ct3:
        test_t3 = st.checkbox(t3_name)
        t3_GP = st.text_input("GP (%):", value = "25", key='t3')
    with ct4:
        test_t4 = st.checkbox(t4_name)
        t4_GP = st.text_input("GP (%):", value = "28", key='t4')
    with ct5:
        test_t5 = st.checkbox(t5_name)
        t5_GP = st.text_input("GP (%):", value = "27", key='t5')
    
    with cs3:
        GP_promo = st.text_input("3+1 Promo GP (%):", value="25",key='t_promo')
    
    df = pd.DataFrame.from_dict(response['selected_rows'])
    col_tier_   = [t1_name, t2_name,t3_name,t4_name,t5_name]#test1_name, test2_name, test3_name]
    col_GP_     = [t1_GP,t2_GP,t3_GP,t4_GP,t5_GP]#,GP_1,GP_2,GP_3]
    col_mask    = [test_t1, test_t2,test_t3,test_t4,test_t5]#[True,True,True,True,True,test_t1, test_t2,test_t3]
    col_tier    = list(np.array(col_tier_)[col_mask])
    col_GP      = list(np.array(col_GP_)[col_mask])
    
    temp_list_ = []
    for input_GP in col_GP:
        temp_list_.append(to_float(input_GP))
    
    captured_values_only = st.sidebar.checkbox("Show captured erroneous values only.")
    st.sidebar.caption('Program may run slow when unchecked. Uncheck before saving for website template.')    
    st.sidebar.markdown("""---""")
    
    if len(temp_list_) != sum(temp_list_):
        st.title("Input Error")
        st.stop()
    
    else:
        if len(df)>0:
            df = df.drop(['dimensions','make'], axis =1).set_index('model')
            df = df.replace('',np.nan).dropna(axis=1, how='all')
            if 'rowIndex' in df.columns.to_list():
                df = df.drop('rowIndex', axis =1)
            if '_selectedRowNodeInfo' in df.columns.to_list():
                df = df.drop('_selectedRowNodeInfo', axis =1)
            df = df_numeric(df)
            column_eval = df.columns.tolist()
            for column in df.columns.tolist():
                if column in ['supplier_max_price','GulongPH','GulongPH_slashed','b2b','marketplace','GoGulong', 'GoGulong_slashed', 'TireManila']:
                    column_eval.remove(column)
            if len(column_eval)>0 and len(selected_supplier_)<len(cols_option):
                df['selection_max_price'] = df[column_eval].fillna(0).apply(lambda x: x.max(),axis=1)
                for c in range(len(col_tier)):
                    df[col_tier[c]] = df['supplier_max_price'].apply(lambda x: consider_GP(x,col_GP[c]))
                if captured_values_only: 
                    df = filter_data_captured(df,col_tier)
                df['3+1_promo_per_tire'] = df['supplier_max_price'].apply(lambda x: promotize(x,GP_promo))
                st.dataframe(df.style.apply(highlight_promo, axis=None).apply(highlight_others,axis=None).apply(highlight_smallercompetitor,axis=None).format(precision = 2))
            else:
                for c in range(len(col_tier)):
                    df[col_tier[c]] = df['supplier_max_price'].apply(lambda x: consider_GP(x,col_GP[c]))
                if captured_values_only: 
                    df= filter_data_captured(df,col_tier)
                df['3+1_promo_per_tire'] = df['supplier_max_price'].apply(lambda x: promotize(x,GP_promo))
                st.dataframe(df.style.apply(highlight_promo, axis=None).apply(highlight_others,axis=None).apply(highlight_smallercompetitor,axis=None).format(precision = 2))
        else:
            st.info("Kindly check/select at least one row above.")
    
    CPC1,CPC2 = st.columns([3,2])
    with CPC1:
        st.write('Showing '+str(len(df))+" out of "+str(len(df_show))+" entries.")
    with CPC2:
        if len(df)>0:
            csvV = convert_df(df)
            st.download_button(label="📥 Download this table as csv",
                                data=csvV,
                                file_name='price_comparison.csv',
                                mime='text/csv')

if edit_mode == 'Automated':
    st.header("Automation parameters")
    with st.expander('Show comparative data:'):
        df_showsummary = df_final_[['make','model','GulongPH','GoGulong','TireManila']]
        
        
        gogulong_i = len(df_showsummary.loc[df_showsummary['GulongPH']>df_showsummary['GoGulong']])
        tiremanila_i = len(df_showsummary.loc[df_showsummary['GulongPH']>df_showsummary['TireManila']])
        a1,a2 = st.columns([1,1])
        with a1:
            st.info("SKUs where GoGulong prices are cheaper than GulongPH: "+str(gogulong_i))
        with a2:
            st.info("SKUs where TireManila prices are cheaper than GulongPH: "+str(tiremanila_i))
        csvComp = convert_df(df_final[['make','model','GulongPH','GoGulong','TireManila']])
        csvCompAdj = convert_df(df_final_[['make','model','GulongPH','GoGulong','TireManila']])
        cAA, cAB = st.columns([1,1])
        with cAA:
            st.download_button(label="📥 Download table below as csv",
                            data=csvComp,
                            file_name='adjusted_price_comparison.csv',
                            mime='text/csv',key='zzz')
        with cAB:
            st.download_button(label="📥 Download raw comparison table as csv",
                            data=csvCompAdj,
                            file_name='raw_price_comparison.csv',
                            mime='text/csv',key='aaa')
        st.table(df_showsummary.style.apply(highlight_smallercompetitor,axis=None).format(precision = 2))
    CA,CB,CC, CD = st.columns([1,1,1,1])
    with CA:
        d_b2b = st.text_input('Set B2B GP:', value = st.session_state['d_b2b'])
        if to_float(d_b2b):
            st.session_state['d_b2b'] = float(d_b2b)
        else:
            d_b2b = st.session_state['d_b2b']
            st.write('Input error, B2B GP set to '+str(d_b2b))
    with CB:
        d_affiliate = st.text_input('Set affiliate GP:', value = st.session_state['d_affiliate'])
        if to_float(d_affiliate):
            st.session_state['d_affiliate'] = float(d_affiliate)
        else:
            d_affiliate = st.session_state['d_affiliate']
            st.write('Input error, affiliate discount set to '+str(d_affiliate))
    with CC:
        d_marketplace = st.text_input('Set marketplace GP:', value = st.session_state['d_marketplace'])
        if to_float(d_marketplace):
            st.session_state['d_marketplace'] = float(d_marketplace)
        else:
            d_marketplace = st.session_state['d_marketplace']
            st.write('Input error, marketplace discount set to '+str(d_marketplace))
    with CD:
        st.button('Apply changes ')
   
    with st.expander("Modify automated pricing rules with respect to GoGulong"):
        
        Ca,Cb,Cc = st.tabs(['Case 1', 'Case 2', 'Case 3'])
        with Ca:
            st.markdown("""
                        #### If GoGulong GP < 15%, then set GulongPH GP to 15%.
                        """)
            
            GP_15_raw = st.text_input('Set GP:', value = st.session_state['GP_15'], help ='Set GulongPH GP to this amount (%)')
            if to_float(GP_15_raw):
                GP_15 = float(GP_15_raw)
                st.session_state['GP_15'] = GP_15
                df_test1 = adjust_wrt_gogulong(df_final_,GP_15=GP_15)
                df_test1 = df_test1.loc[df_test1['GoGulong_GP']<15]
                if len(df_test1) >0:
                    e1a, e1b = st.columns([1,5])
                    with e1a:
                        show_15 = st.checkbox('Show all', key = 'gp15')
                    
                    if show_15:
                        df_show_15 = df_test1[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].set_index('model')
                        text = 'Showing all:'
                    
                    else:
                        text = 'Example:'
                        df_show_15 = df_test1[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].head().set_index('model')
                    with e1b:
                        st.write('Example:')
                    st.dataframe(df_show_15.style.format(precision = 2))
                    st.caption('Showing '+str(len(df_show_15))+' of '+str(len(df_test1))+' changes.')
            else:
                GP_15 = 15.
                st.write('Input error, GP set to 15%')
        with Cb:
            st.markdown("""
                        #### If GoGulong GP is between 15% and 20%, match GulongPH price.
                        """)
            
            GP_20_raw = st.text_input('Price offset value: ', value = st.session_state['GP_20'], help='Decrease GoGulong by this amount (Php)')
            if to_float(GP_20_raw):
                GP_20 = float(GP_20_raw)
                st.session_state['GP_20'] = GP_20
                df_test2 = adjust_wrt_gogulong(df_final_,GP_20=GP_20)
                df_test2 = df_test2.loc[(df_test2['GoGulong_GP']<20) & (df_test2['GoGulong_GP']>=15)]
                if len(df_test2) >0:
                    e2a, e2b = st.columns([1,5])
                    with e2a:
                        show_20 = st.checkbox('Show all', key = 'gp20')
                    
                    if show_20:
                        df_show_20 = df_test2[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].set_index('model')
                        text = 'Showing all:'
                    
                    else:
                        text = 'Example:'
                        df_show_20 = df_test2[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].head().set_index('model')
                    with e2b:
                        st.write('Example:')
                    st.dataframe(df_show_20.style.format(precision = 2))
                    st.caption('Showing '+str(len(df_show_20))+' of '+str(len(df_test2))+' changes.')
            else:
                GP_20 = 5.
                st.write('Input error, price offset is floor_5')
        with Cc:
            st.markdown("""
                        #### If GoGulong GP > 20%, then adjust GP correspondingly.
                        """)
            
            GP_20__raw = st.text_input('GP Offset value: ', value = st.session_state['GP_20_'], help = 'Decrease GoGulong GP by this amount (%)')
            if to_float(GP_20__raw):
                GP_20_ = float(GP_20__raw)
                st.session_state['GP_20_'] = GP_20_
                df_test3 = adjust_wrt_gogulong(df_final_,GP_20_=GP_20_)
                df_test3 = df_test3.loc[df_test3['GoGulong_GP']>=20]
                if len(df_test3) >0:
                    e3a, e3b = st.columns([1,5])
                    with e3a:
                        show_20_ = st.checkbox('Show all', key = 'gp20_')
                    
                    if show_20_:
                        df_show_20_ = df_test3[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].set_index('model')
                        text = 'Showing all:'
                    
                    else:
                        text = 'Example:'
                        df_show_20_ = df_test3[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].head().set_index('model')
                    with e3b:
                        st.write('Example:')
                    st.dataframe(df_show_20_.style.format(precision = 2))
                    st.caption('Showing '+str(len(df_show_20_))+' of '+str(len(df_test3))+' changes.')
            else:
                GP_20_ = 1.
                st.write('Input error, GP offset is floor_3')
        st.button('Apply changes')
    if st.session_state['adjusted']: 
        st.write("Prices have been adjusted. The the following Gulong.ph SKU prices are still greater than GoGulong's SKU prices.")
    else:
        st.write("Kindly review the changes that would be implemented to the product prices.")
    
    
    df_adjusted = adjust_wrt_gogulong(df_final_,GP_15,GP_20,GP_20_,st.session_state['d_b2b'],st.session_state['d_affiliate'],st.session_state['d_marketplace'])
    if len(df_adjusted) >0:
        
        df_temmmp =df_final_[['model', 'supplier_max_price','GulongPH']].merge(df_adjusted[['model','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP','GulongPH_slashed','b2b','affiliate','marketplace']], on='model',suffixes = ('_backend','_adjusted'), how='right').set_index('model')
        st.dataframe(df_temmmp.style.format(precision = 2))#lsuffix = '_backend', rsuffix='_adjusted'
        cRa, cRb = st.columns([2,1])
        with cRa:
            st.caption("Reviewing "+ str(len(df_temmmp))+" changes to be implemented out of "+ str(len(df_final_.loc[df_final_['GoGulong'].notnull()]))+" SKU overlaps.")
        
        with cRb:
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df_temmmp.reset_index().to_excel(writer, index=False, sheet_name='All changes')
            df_test1[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 1')
            df_test2[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 2')
            df_test3[['model','supplier_max_price','GulongPH','GoGulong','GulongPH_GP','GoGulong_GP']].to_excel(writer, index=False, sheet_name='Case 3')
            workbook = writer.book
            writer.save()
            processed_data = output.getvalue()
            st.download_button(label='📥 Download this data',
                                        data=processed_data ,
                                        file_name= 'automated_changes.xlsx')
    else:
        st.info('No changes to implement.')
    
st.markdown("""
            ---
            """)
            
st.header('Download tables:')
st.write("**Backend data:**")
if df_final is not None:
    csvA = convert_df(df_final)
    st.download_button(
                        label="backend_data.csv",
                        data=csvA,
                        file_name='backend_data.csv',
                        mime='text/csv'
                        )
if edit_mode == 'Manual':
    st.write("**All data:**")
    if df_show is not None:
        csvB = convert_df(df_show)
        st.download_button(label="gulong_pricing.csv",
                            data=csvB,
                            file_name='gulong_pricing.csv',
                            mime='text/csv')

# if len(temp_list_) != sum(temp_list_):
#     st.title("Input Error")
#else:


st.write("**Final data:**")        
final_columns =                     ['Brand', 'TIRE_SKU', 
        'Section_Width', 'Height', 'Rim_Size', 'Pattern',
        'LOAD_RATING', 'SPEED_RATING', 'Supplier_Price', 'SRP', 'PROMO_PRICE',
        'b2b_price', 'mp_price', 'STOCKS_ON HAND', 'Primary_Supplier']
df_final_ = df_final_.set_index('model')#.reset_index() [['make','model',
        # 'section_width', 'aspect_ratio','rim_size','pattern',
        # 'load_rating', 'speed_rating','supplier_max_price','GulongPH_slashed','GulongPH',
        # 'b2b','marketplace''stock','supplier_id']]
if edit_mode =='Manual' and len(df)>0:
    req_cols = ['model','supplier_max_price','GulongPH_slashed','GulongPH','b2b','marketplace']
    req_cols.extend(col_tier)
    final_ = df.reset_index()[req_cols].set_index('model')
    df_final_.update(final_)

final_df = df_final_.reset_index()
final_df['aspect_ratio'] = final_df['aspect_ratio'].fillna(0)
final_df['speed_rating'] = final_df['speed_rating'].fillna(' ')       
final_df = pd.concat([final_df.set_index('model'),df_final_],axis=0)
final_df = final_df[~final_df.index.duplicated(keep='first')].reset_index().sort_values(by='model')[['make','model',
        'section_width', 'aspect_ratio','rim_size','pattern',
        'load_rating', 'speed_rating','supplier_max_price','GulongPH_slashed','GulongPH',
        'b2b','marketplace','stock','supplier_id']]

df_temmmp_ =df_final[['model','supplier_max_price','GulongPH_slashed','GulongPH','b2b','marketplace']].merge(df_final_.reset_index()[['model','GulongPH_slashed','GulongPH','b2b','marketplace']], on='model',suffixes = ('_backend','_adjusted'), how='left')
with st.expander('Review changes in prices'):
    r1,r2,r3,r4 = st.tabs(['SRP', 'Promo Price', 'B2B Price', 'Marketplace'])
    with r1:
      st.dataframe(df_temmmp_.loc[df_temmmp_['GulongPH_slashed_backend']!=df_temmmp_['GulongPH_slashed_adjusted'],['model','supplier_max_price','GulongPH_slashed_backend','GulongPH_slashed_adjusted']].set_index('model').style.format(precision = 2))
    with r2:
      st.dataframe(df_temmmp_.loc[df_temmmp_['GulongPH_backend']!=df_temmmp_['GulongPH_adjusted'],['model','supplier_max_price','GulongPH_backend','GulongPH_adjusted']].set_index('model').style.format(precision = 2))
    with r3:
      st.dataframe(df_temmmp_.loc[df_temmmp_['b2b_backend']!=df_temmmp_['b2b_adjusted'],['model','supplier_max_price','b2b_backend','b2b_adjusted']].set_index('model').style.format(precision = 0))
    with r4:
      st.dataframe(df_temmmp_.loc[df_temmmp_['marketplace_backend']!=df_temmmp_['marketplace_adjusted'],['model','supplier_max_price','marketplace_backend','marketplace_adjusted']].set_index('model').style.format(precision = 0))


final_df.columns = final_columns
if edit_mode =='Manual':
    for c in range(len(col_tier)):
        final_df[col_tier[c]] = final_df['Supplier_Price'].apply(lambda x: consider_GP(x,col_GP[c]))
    final_df['3+1_promo_per_tire'] = final_df['Supplier_Price'].apply(lambda x: promotize(x,GP_promo))
    st.error('Make sure that the cells that have been changed are included in the selected cells in the pivot table.\nMake sure that the checkbox for "Show captured erroneous values only" is unchecked.')
    

df_xlsx = to_excel(final_df)
st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'website_template.xlsx')

st.sidebar.caption('Last updated on 2023/04/12')

st.markdown("""
            ---
            """)