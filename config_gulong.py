# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:21:36 2023

@author: carlo
"""
import pandas as pd
import numpy as np
from decimal import Decimal
import re
from pytz import timezone
from fuzzywuzzy import fuzz, process
import doctest

# set timezone
phtime = timezone('Asia/Manila')

def import_makes():
    '''
    Import list of makes
    '''
    with open("gulong_makes.txt") as makes_file:
        makes = makes_file.readlines()
        
    makes = [re.sub('\n', '', m).strip() for m in makes]
    return makes

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

def fix_names(sku_name, comp=None):
    '''
    Fix product names to match competitor names
    
    Parameters
    ----------
    sku_name: str
        input SKU name string
    comp: list (optional)
        optional list of model names to compare with
    
    Returns
    -------
    name: str
        fixed names as UPPERCASE
    '''
    
    # replacement should be all caps
    change_name_dict = {'TRANSIT.*ARZ.?6-X' : 'TRANSITO ARZ6-X',
                        'TRANSIT.*ARZ.?6-A' : 'TRANSITO ARZ6-A',
                        'TRANSIT.*ARZ.?6-M' : 'TRANSITO ARZ6-M',
                        'OPA25': 'OPEN COUNTRY A25',
                        'OPA28': 'OPEN COUNTRY A28',
                        'OPA32': 'OPEN COUNTRY A32',
                        'OPA33': 'OPEN COUNTRY A33',
                        'OPAT\+': 'OPEN COUNTRY AT PLUS', 
                        'OPAT2': 'OPEN COUNTRY AT 2',
                        'OPMT2': 'OPEN COUNTRY MT 2',
                        'OPAT OPMT': 'OPEN COUNTRY AT',
                        'OPAT': 'OPEN COUNTRY AT',
                        'OPMT': 'OPEN COUNTRY MT',
                        'OPRT': 'OPEN COUNTRY RT',
                        'OPUT': 'OPEN COUNTRY UT',
                        'DC -80': 'DC-80',
                        'DC -80+': 'DC-80+',
                        'KM3': 'MUD-TERRAIN T/A KM3',
                        'KO2': 'ALL-TERRAIN T/A KO2',
                        'TRAIL-TERRAIN T/A' : 'TRAIL-TERRAIN',
                        '265/70/R16 GEOLANDAR 112S': 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112S' : 'GEOLANDAR A/T G015',
                        '265/65/R17 GEOLANDAR 112H' : 'GEOLANDAR G902',
                        'GEOLANDAR A/T 102S': 'GEOLANDAR A/T-S G012',
                        'GEOLANDAR A/T': 'GEOLANDAR A/T G015',
                        'ASSURACE MAXGUARD SUV': 'ASSURANCE MAXGUARD SUV',
                        'EFFICIENTGRIP SUV': 'EFFICIENTGRIP SUV',
                        'EFFICIENGRIP PERFORMANCE SUV':'EFFICIENTGRIP PERFORMANCE SUV',
                        'WRANGLE DURATRAC': 'WRANGLER DURATRAC',
                        'WRANGLE AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT ADVENTURE': 'WRANGLER AT ADVENTURE',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'ENASAVE  EC300+': 'ENSAVE EC300 PLUS',
                        'SAHARA AT2' : 'SAHARA AT 2',
                        'SAHARA MT2' : 'SAHARA MT 2',
                        'WRANGLER AT SILENT TRAC': 'WRANGLER AT SILENTTRAC',
                        'POTENZA RE003 ADREANALIN': 'POTENZA RE003 ADRENALIN',
                        'POTENZA RE004': 'POTENZA RE004',
                        'SPORT MAXX 050' : 'SPORT MAXX 050',
                        'DUELER H/T 470': 'DUELER H/T 470',
                        'DUELER H/T 687': 'DUELER H/T 687 RBT',
                        'DUELER A/T 697': 'DUELER A/T 697',
                        'DUELER A/T 693': 'DUELER A/T 693 RBT',
                        'DUELER H/T 840' : 'DUELER H/T 840 RBT',
                        'EVOLUTION MT': 'EVOLUTION M/T',
                        'BLUEARTH AE61' : 'BLUEARTH XT AE61',
                        'BLUEARTH ES32' : 'BLUEARTH ES ES32',
                        'BLUEARTH AE51': 'BLUEARTH GT AE51',
                        'COOPER STT PRO': 'STT PRO',
                        'COOPER AT3 LT' : 'AT3 LT',
                        'COOPER AT3 XLT' : 'AT3 XLT',
                        'A/T3' : 'AT3',
                        'ENERGY XM+' : 'ENERGY XM2+',
                        'XM2+' : 'ENERGY XM2+',
                        'AT3 XLT': 'AT3 XLT',
                        'ADVANTAGE T/A DRIVE' : 'ADVANTAGE T/A DRIVE',
                        'ADVANTAGE T/A SUV' : 'ADVANTAGE T/A SUV'
                        }
    
    if pd.isna(sku_name) or (sku_name is None):
        return np.NaN
    
    else:
        # uppercase and remove double spaces
        raw_name = re.sub('  ', ' ', sku_name).upper().strip()
        # specific cases
        for key in change_name_dict.keys():
            if re.search(key, raw_name):
                return change_name_dict[key]
            else:
                continue
        
        # if match list provided
        
        if comp is not None:
            # check if any name from list matches anything in sku name
            match_list = [n for n in comp if re.search(n, raw_name)]
            # exact match from list
            if len(match_list) == 1:
                return match_list[0]
            # multiple matches (i.e. contains name but with extensions)
            elif len(match_list) > 1:
                long_match = ''
                for m in match_list:
                    if len(m[0]) > len(long_match):
                        long_match = m[0]
                return long_match
            # no match
            else:
                return raw_name
        else:
            return raw_name

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
    >>> combine_specs('175', '65', '15', mode = 'SKU')
    '175/65/R15'
    
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
        w = ''.join(re.findall('[0-9]|\.', str(w)))
        ar = ''.join(re.findall('[0-9]|\.|R', str(ar)))
        d = ''.join(re.findall('[0-9]|\.', str(d)))
        return '/'.join([w, ar, d])

    else:
        combine_specs(str(w), str(ar), str(d), mode = 'SKU')
   
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

def clean_width(w, model = None):
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
    >>> clean_width('7')
    '7'
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
        if model is None:
            return np.NaN
        else:
            try:
                width = model.split('/')[0].split(' ')[-1].strip().upper()
                return clean_width(width)   
            except:
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
        # incorrect parsing osf decimal aspect ratios
        elif str(ar) in error_ar.keys():
            return error_ar[str(ar)]
        # numeric/integer aspect ratio
        elif str(ar).isnumeric():
            return str(ar)
        # decimal aspect ratio with trailing 0
        else:
            return str(remove_trailing_zero(Decimal(str(ar))))
    else:
        return 'R'
    
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
    
def combine_sku(make, w, ar, d, model, load, speed):
    '''
    DOCTESTS:
            
    >>> combine_sku('ARIVO', '195', 'R', 'R15', 'TRANSITO ARZ 6-X', '106/104', 'Q')
    'ARIVO 195/R15 TRANSITO ARZ 6-X 106/104Q'
    
    '''
    specs = combine_specs(w, ar, d, mode = 'SKU')
    
    try:
        SKU = ' '.join([make, specs, model])
    except:
        SKU = ' '.join([make, specs])
    
    finally:
        if (load in ['nan', np.NaN, None]) and (speed in ['nan', np.NaN, None]):
            pass
        else:
            SKU = SKU + ' ' + load + speed
        return SKU
    