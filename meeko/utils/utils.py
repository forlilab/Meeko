#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
# collection of useful snippets of code that's used frequently
#

import os
import re
import sys
import importlib


def path_module(module_name):
    try:
        from importlib import util
        specs = util.find_spec(module_name)
        if specs is not None:
            return specs.submodule_search_locations[0]
    except:
        try:
            _, path, _ = imp.find_module(module_name)
            abspath = os.path.abspath(path)
            return abspath
        except ImportError:
            return None
    return None

def getNameExt(fname):
    """ extract name and extension from the input file, removing the dot
        filename.ext -> [filename, ext]
    """
    name, ext = os.path.splitext(fname)
    return name, ext[1:] #.lower()


# def writeList(filename, inlist, mode = 'w', addNewLine = False):
#     if addNewLine: nl = "\n"
#     else: nl = ""
#     fp = open(filename, mode)
#     for i in inlist:
#         fp.write(str(i)+nl)
#     fp.close()


#def getResInfo(string):
#    """ CHAIN:RESnum -> [ "CHAIN", "RES", num ]"""
#    if ':' in string:
#        chain, resraw = string.split(':')
#    else:
#        chain = ''
#        resraw = string
#    try:
#        res = resraw[0:3]
#        num = int(resraw[3:])
#    except:
#        # heuristic for nucleic acids
#        regex = r'[UACGT]+\d'
#        match = re.search(regex, resraw)
#        if match is None:
#            print("WARNING! Unknown residue naming scheme")
#            return chain, "X", "X"
#        res = resraw[0]
#        num = int(resraw[1:])
#        #print "NUCLEIC:",  chain, res, num
#    return chain, res, num


# def get_data_file(file_handle, dir_name, data_file):
#     module_dir, module_fname = os.path.split(file_handle)
#     DATAPATH = os.path.join(module_dir, dir_name, data_file)
#     return DATAPATH

