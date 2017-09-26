# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:49:13 2017

@author: kapot
"""

import datetime
from os import path
from zipfile import ZipFile

import dfparser
import pandas as pd

gauth = None
drive = None

def __init_grive():
    global gauth
    global drive
    
    if not gauth:
        from pydrive.auth import GoogleAuth
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
    
    if not drive:
        from pydrive.drive import GoogleDrive
        drive = GoogleDrive(gauth)
    

def get_points_drom_drive():
    __init_grive()
    points_raw = drive.ListFile({'q': "title contains '.rsb.zip'"}).GetList()

    times = [datetime.datetime.strptime(p["originalFilename"], \
             "%Y%m%d-%H%M%S.rsb.zip") for p in points_raw]

    pd.set_option('display.max_columns', 500)
    points = pd.DataFrame({'name': [p["originalFilename"] for p in points_raw],
                           'time': times,
                           'id': [p["id"] for p in points_raw],
                           'size': [int(p["fileSize"]) for p in points_raw]})
    return points


def load_dataset(id, name, size=0):
    
    """
      Скачать файл данных платы ЛАН10-12PCI с диска и открыть его парсером
      @id - идентификатор файла на гугл диске
      @name - исходное имя файла на гугл диске
      @size - размер файла в байтах на диске. Используется для проверки 
      скачанного файла
      @return - открытый датасет
      
    """
    
    __init_grive()
    if not(path.exists(name) and path.getsize(name) == size):
        file = drive.CreateFile({"id": id})
        file.GetContentFile(name)
    
    extracted_name = path.basename(name[:-4])
    
    if not path.exists(extracted_name):
        unpacked = ZipFile(name, "r")
        extracted_name = unpacked.extract(path.basename(name[:-4]))

    dataset =  dfparser.RshPackage(extracted_name)
    return dataset


