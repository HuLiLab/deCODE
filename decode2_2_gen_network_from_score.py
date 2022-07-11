from __future__ import absolute_import
from __future__ import print_function
import time
import sys
import os
import re
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt    


sys.path.append('/home/m143944/pharmacogenomics/gene_drug_2d/6_vis_weight')
import lib_cnn_routine

np.random.seed(1337)  # for reproducibility
def append_to_mainfn(fnin, str_app):
  (strpath, strfn) = os.path.split(os.path.abspath(fnin))
  (strmainfn, strext) = os.path.splitext(strfn)

  (strmainfn2, strext2) = os.path.splitext(strmainfn)#secondary ext
  if strmainfn2=='' or strext2=='':#secondary split not needed
    pass
    #strmainfn=strmainfn; strext=strext#no operation
  else:#secondary splittinig needed
    strmainfn=strmainfn2
    strext=strext2+strext
  str_r = os.path.join(strpath, strmainfn+ str_app +strext)
  return str_r
#end def append_to_mainfn(fnin, str_app):


def float_to_str(v):
  #make a float value suitable for file name by removing its decimal dot.
  #
  s='%g'%v
  if '.' in s: s=s.replace('.','')
  return s

########################################################################################################
#def main():
  
nargs=len(sys.argv)
if nargs!=2:
  print('usage: '+os.path.basename(__file__)+' <configuration file>')
  sys.exit()
#end if
fpcfg=os.path.abspath(sys.argv[1])#'tempcfg.cfg'
fntrunc=os.path.splitext(fpcfg)[0]
wd=os.path.dirname(fpcfg)
print('input file: %s'%fpcfg)
print('changing cwd to folder %s'%wd)
os.chdir(wd)
ts=time.time()

fnh5=fntrunc+'.rawYBscore.h5'
#print('writing to file ', fnh5)
#store = pd.HDFStore(fnh5)
#store['default']=pd_s
#store.close()

print('loading key default from file ', fnh5)
store = pd.HDFStore(fnh5)
pd_s=store['default']
store.close()
print('check pd_s')
#sys.exit()

#  #flatten?
#  #coordinates of flattened matrix: 
#  mt=np.asarray([[11, 12],[21,22],[31, 32]])                            
#  mt
#  #array([[11, 12],
#  #        [21, 22],
#  #        [31, 32]])
#  mtf=mt.flatten()
#  #array([11, 12, 21, 22, 31, 32])
#  ncol=2
#  # pos in flattened matrix to i, j
#  # i=floor(pos/ncol)
#  # j=pos % ncol
#  mtrec=np.zeros((len(mtf)/ncol,ncol), dtype=int)
#  for pos in range(len(mtf)):
#    i=int(np.floor(pos/ncol))
#    j=pos%ncol
#    mtrec[i,j]=mtf[pos]
#  #end for
#  mtrec
#  #
#  #array([[11, 12],
#  #       [21, 22],
#  #       [31, 32]])


ts=time.time()
print('flatten start')
#mf=pd_s.as_matrix().flatten()
mf=pd_s.values.flatten()

print('argsort')
ret=np.argsort(-np.abs(mf))#the sort was ascendingly. to do descending, add -1
ret[:3]
mf[ret[:3]]
ncol=pd_s.shape[1]
#ntop=100#select top 200 edges
#print('extracting ntop %d'%ntop)
#for pos in ret[:ntop]:
#  i=int(np.floor(pos/ncol))
#  j=pos%ncol
#  val=mf[pos]
#  val2=pd_s.iloc[i,j]
#  gfrom=pd_s.index[i]
#  gto=pd_s.columns[j] 
#  print('%s -> %s %f'%(gfrom, gto, val))
#  if val!=val2: 
#    print('val=%f, val2=%f'%(val, val2))
#    raw_input()

#excluding self loop
#ntop=50#500#select edges with top abs weights
ntop=200#500#select edges with top abs weights

print('extracting ntop %d'%ntop)
nsel=0
ret_i=0
ll=list()
while nsel<ntop:
  pos=ret[ret_i]
  i=int(np.floor(pos/ncol))
  j=pos%ncol
  ret_i+=1
  if i==j: continue
  nsel+=1 
  val=mf[pos]
  val2=pd_s.iloc[i,j]
  gfrom=pd_s.index[i]
  gto=pd_s.columns[j] 
  print('%s -> %s %f'%(gfrom, gto, val))
  ll.append([gfrom, gto, float(val)])
  if val!=val2: 
    print('val=%f, val2=%f'%(val, val2))
    raw_input()
#end while

#save edge list to gml using networkx
#fnouttrunc=os.path.splitext(os.path.basename(fnh5))[0]
fnouttrunc=os.path.splitext(fnh5)[0]

fngml=fnouttrunc+'.ntop%d.gml'%ntop
fntxt=fnouttrunc+'.ntop%d.txt'%ntop

g=nx.DiGraph()
for l in ll:
  n1, n2, w=l
  g.add_edge(n1, n2, size=np.abs(w), weight=w)
#end for l
print('writing to file %s'%fngml)
nx.write_gml(G=g, path=fngml)
print('writing to file %s'%fntxt)
nx.write_edgelist(G=g, path=fntxt, delimiter='\t')

print('Done. total time used: %f'%(time.time()-ts))

#return
#end of main()

#if __name__=='__main__': main()

