# Utility functions

import os
import math
import numpy as np
import datetime
import time
import winsound
import sys
from urllib import request
import shutil

import os
import math
import time
import logging
import warnings
import numpy as np
import datetime
from getpass import getpass
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from scipy.optimize import curve_fit, leastsq
from progress.bar import IncrementalBar
from tqdm import tqdm
import winsound

from quantuminspire.api import QuantumInspireAPI
from quantuminspire.version import __version__ as quantum_inspire_version
from quantuminspire.credentials import load_account, get_token_authentication, get_basic_authentication, save_account

def PrepFile(basename: str="", suffix: str="", doraw: int=0):
    '''
    Creates the file name according to the basename and suffix that you provide.
    '''
    histname="Hist_"+basename
    cqasmname="cQASM_"+basename
    rawname="Raw_"+basename
    if (len(suffix)>0):
        histname+='_'+suffix
        cqasmname+='_'+suffix
        rawname+='_'+suffix
    histname+='_API.txt'
    cqasmname+='_API.txt'
    rawname+="_API"

    file=open(histname,'w')
    file=open(cqasmname,'w')
    
    if (doraw==0):
        return histname, cqasmname
    else:
        return histname, cqasmname, rawname


def GetTimeStamp():
    '''
    Returns the timestamp of the current date and time
    '''
    current_date = datetime.datetime.now()
    thisyear=str(current_date.year);
    thismonth="0"+str(current_date.month);
    thisday="0"+str(current_date.day);
    thishour="0"+str(current_date.hour);
    thisminute="0"+str(current_date.minute);
    thissecond="0"+str(current_date.second);
    timestamp=thisyear[-2:]+thismonth[-2:]+thisday[-2:]+"_"+thishour[-2:]+thisminute[-2:]+thissecond[-2:]
    return timestamp

def NewDay():
    '''
    Crates the folder with the current date in the specified path.
    '''
    todaypath = r'' # Write here the path of the folder you want to save your data in

    current_date = datetime.date.today()
    thisyear=str(current_date.year);
    thismonth="0"+str(current_date.month);
    thisday="0"+str(current_date.day);
    todaypath+="/Data_"+thisyear[-2:]+thismonth[-2:]+thisday[-2:]

    try:
        os.mkdir(todaypath)
    except OSError:
        print ("Creation of the directory %s failed" % todaypath)
    else:
        print ("Successfully created the directory %s " % todaypath)
    
    os.chdir(todaypath)  # change the current working directory to the specified path
    return todaypath

def API_RunAndSave(param, qasmprog: str = "foo", histname: str="hist.txt", cqasmname: str="cqasm.txt", shots: int=16384, backend: int=1, lastonly: int=0, getresults: int=0):
    '''
    Runs QI with qasmprog and appends measurement histogram(s) to file histname
    A copy of the cqasm program is saved to file cqasmname.

    param:      a reference number that you are free to choose. it will be saved in the first column of the file
    shots:      desired number of shots. For Starmon-5, the max is 16384.
    backend:    0: QX simulation
                1: Starmon-5
                2: Spin-2
    lastonly:   0: save histograms for all measurement blocks in the circuit
                1: save only the hisogram of the last measurement block in the circuit
    getresults: 0: do not return the results
                1: return the results
    '''
    
    # Set the backend
    QI = QuantumInspireAPI(project_name="APIProject")
    if (backend==0):
        backend_type = QI.get_backend_type_by_name('QX single-node simulator')
    if (backend==1):
        backend_type = QI.get_backend_type_by_name('Starmon-5')
    if (backend==2):
        backend_type = QI.get_backend_type_by_name('Spin-2')
    
    # save cqasm program to file
    file=open(cqasmname,'a')
    file.write(qasmprog)
    file.write('\n\n')
    file.close()

    results = QI.execute_qasm(qasmprog, backend_type=backend_type, number_of_shots=shots)

    numhistos=len(results['histogram'])
    
    starthisto=0
    if(lastonly==1):
        starthisto=numhistos-1

    # record histograms
    for i in range(starthisto, numhistos,1):
        
        # prepare histogram
        counts = np.zeros(2**5)
        histo=results['histogram'][i]
        for thisvaluestr, thisprob in histo.items():
            intval=int(thisvaluestr)
            counts[intval] = thisprob       
        
        # append histogram to file
        file=open(histname,'a')
        file.write('{}\t'.format(param))
        for value in counts:
            file.write('{:.5f}\t'.format(value))
        file.write('\n')
        file.close()

    if (getresults==1):
        return results

def API_SaveRawData(results, param, basename):
    '''
    This function saves the raw data from an API run to a csv file
    results: is what API_RunandSave returns
    param: parameter that you are measuring. It will be added to the final data file name
    basename: name that you want to give to the data file.
    '''
    # some important variables
    api_version = '2.0'
    file_format = "csv"
    output = "console"

    # get url from results
    url=results['raw_data_url']
    url += f"?&format={file_format}"

    # dump data into a temp file
    opener = request.build_opener()
    opener.addheaders = [('accept', f'application/coreapi+json, application/vnd.coreapi+json, */*; version={api_version};')]
    request.install_opener(opener)
    local_filename, headers = request.urlretrieve(url, filename=None if output == "console" else output)

    # create directory if it doesn't yet exist
    if(os.path.isdir(basename)==False):
        os.mkdir(basename)

    # specify destination file
    dest = basename+r"/"+basename+"_"+str(param)+".csv"
    
    # copy from temp file to destination file
    path = shutil.copyfile(local_filename,dest)


def DeleteProjects(basename: str=""):
    '''
    Deletes the specified project
    '''
    QI = QuantumInspireAPI()
    for pi in QI.get_projects():
        if basename in pi['name']:
            QI.delete_project(pi['id'])
    
    frequency = 100  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    return   
