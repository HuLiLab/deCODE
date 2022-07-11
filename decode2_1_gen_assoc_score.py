from __future__ import absolute_import
from __future__ import print_function
import time
import sys
import os
import re
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import lib_decode

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
print(sys.argv)
if nargs!=2 and nargs!=3:
  print('usage: '+os.path.basename(__file__)+' <configuration file> [filtered_gene_list.txt]')
  print('generate raw YB score from model weights; filter by the gene list if present.')
  sys.exit()
#end if
gl=None
if nargs==3:
  fngl=sys.argv[2]
  gl=list()
  for l in open(fngl,'rb'):
    ls=l.strip()
    if ls=='': continue
    gl.append(ls)
  #end for
#end if
fpcfg=os.path.abspath(sys.argv[1])#'tempcfg.cfg'
fntrunc=os.path.splitext(fpcfg)[0]
cmi=lib_decode.cnn_model_info(fpcfg)
l_nndense=cmi.d['nndense'].split('-')
if len(l_nndense)%2==0:#even number of dense layers, no middle layer
  print('cannot find middle layer with even number of dense layers. exit.')
  print(l_nndense)
  sys.exit()
#end if
mid_lyr_idx=(len(l_nndense)+1)/2-1
print('mid_lyr_idx=%d'%mid_lyr_idx)

wd=os.path.dirname(fpcfg)
print('input file: %s'%fpcfg)
print('changing cwd to folder %s'%wd)
os.chdir(wd)
ts=time.time()
fn_train=cmi.d['train_data_file']
print('loading traning samples from ', fn_train)
print('before loading: %f M'%lib_decode.memory_usage_resource())
if ';' not in fn_train and ( fn_train.endswith('.gz') or fn_train.endswith('.txt') ):
  pass
else:
#  print('training data should be single tab-delimited text file.')
  print('using the 1st file in the training file list for gene names.')
  fn_train=fn_train.split(';')[0]
#  sys.exit()
#end if
df=pd.read_csv(fn_train, sep='\t', header=0, index_col=0, compression='infer')
#df.shape
#(11840, 99)
feature_names=list(df.index)

print('keras: using functional api to build model instead of graph')
import keras
from keras.models import Model
model=None
epoch_trained=0
[model, epoch_trained]=cmi.try_load_model_with_weights()
print('model loaded.')

print('time used so far: %.2f seconds.'%(time.time()-ts))
#sys.exit()
'''
In [5]: model.layers
Out[5]: 
[<keras.engine.topology.InputLayer at 0x7f98dee57e10>,
 <keras.layers.core.Dense at 0x7f98dee57fd0>,
 <keras.layers.core.Dense at 0x7f98dedf7050>]
'''
#InputLayer has no weight
'''
In [67]: model.layers[1].get_weights()[0].shape
Out[67]: (11840, 50)
In [68]: model.layers[1].get_weights()[1].shape                                
Out[68]: (50,)
'''
idx_dense1=None
idx_dense2=None
l_idx_dense=list()
for i in range(len(model.layers)):
  if '.Dense' in str(model.layers[i]):
    l_idx_dense.append(i)
#end for i
idx_dense1=l_idx_dense[0]
idx_dense2=l_idx_dense[1]

#sys.exit()

##################################################################
#feature selection using method 5.1.1 in review paper doi 10.1.1.54.4570
#
#  original paper for the method: 
#Yacoub,  M.  and  Bennani,  Y.  (1997).  
#HVS:  A  Heuristic  for  Variable  Selection  in  Multilayer Artificial Neural Network Classifier.
#in Proceedings of ANNIE'97. 527-532
#
#
#  for neural network I->H->O
#  where I is the input layer, O is the output layer, H is the hidden layer
#
#S[i] = sum[o in O] sum [j in H] abs( wn[o,j]*wn[j,i] )
#
#where S[i] is the importance score for input feature i (YB score)
#  wn[o,j] and wn[j,i] are normalized weights. 
#  defined as 
#  wn[o,j]=w[o,j]/sum(abs(w[o,.]))
#  w[o,j] is the original weight and w[o,.] are the weights of output neuron o to all neurons in H.
#
#w1abs=np.abs(w1)
#w2abs=np.abs(w2)
#
#w1abssum=np.apply_along_axis(func1d=lambda a: np.sum(a), axis=0, arr=w1abs)#w1 abs sum, the L1 norm of w1
#print('w1abssum.shape=', w1abssum.shape)
#
#w2abssum=np.apply_along_axis(func1d=lambda a: np.sum(a), axis=0, arr=w2abs)#w2 abs sum, the L1 norm of w2
#print('w2abssum.shape=', w2abssum.shape)
##sys.exit()
#
#nfeat=w1.shape[0]
#nhidden=w2.shape[0]#also w1.shape[1]
#if nhidden!=w1.shape[1]:#error
#  print('error, w1.shape[1]=%d but w2.shape[0]=%d, not equal. exit.'%(w1.shape[1], w2.shape[0]))
#  sys.exit()
##end if
#noutput=w2.shape[1]
#print('nfeat=%d, nhidden=%d, noutput=%d'%(nfeat, nhidden, noutput))
##normalize
#w1absnorm=np.copy(w1abs)
#w2absnorm=np.copy(w2abs)
#
#for j in range(nhidden):
#  w1absnorm[:,j]/=w1abssum[j]
#for k in range(noutput):
#  w2absnorm[:,k]/=w2abssum[k]
###########################################################

s0raw=None
s0=None
i=0
smid=None
smidraw=None
for idxd in l_idx_dense:
  print('layer %d: '%idxd, end=' ')
  w_i=model.layers[idxd].get_weights()[0]
  print(w_i.shape) 
  wi_abs=np.abs(w_i)
  wiabssum=np.apply_along_axis(func1d=lambda a: np.sum(a), axis=0, arr=wi_abs)#w1 abs sum, the L1 norm of w1
  print('wiabssum.shape=', wiabssum.shape)
  wiabsnorm=np.copy(wi_abs)

  for j in range(wiabsnorm.shape[1]):
    wiabsnorm[:,j]/=wiabssum[j]

  if type(s0raw)==type(None):
    s0raw=w_i
    s0=wiabsnorm
  else:
    s0raw=np.dot(s0raw, w_i)
    s0=np.dot(s0, wiabsnorm)
  #end if
  if idxd==mid_lyr_idx:#
    smid=np.copy(s0)
    smidraw=np.copy(s0raw)
  #end if
  i+=1 
#layer 0:  (33878, 3000)
#layer 1:  (3000, 50)
#layer 2:  (50, 3000)
#layer 3:  (3000, 33878)
#result (33878, 33878), 1st dim for input (row), 2nd dim for output (column)
#sys.exit()
#
#s0raw=np.dot(w1, w2)#no abs, no normalization
  #end for
pd_s=pd.DataFrame(s0raw, index=feature_names, columns=feature_names)
print('check pd_s')
#sys.exit()
fnh5=fntrunc+'.rawYBscore.h5'
print('writing to file ', fnh5)
store = pd.HDFStore(fnh5)
store['default']=pd_s
store.close()
print('time used so far: %.2f seconds.'%(time.time()-ts))

pd_s_n=pd.DataFrame(s0, index=feature_names, columns=feature_names)#normalized
print('check pd_s_n')
#sys.exit()
fnh5=fntrunc+'.YBscore.h5'
print('writing to file ', fnh5)
store = pd.HDFStore(fnh5)
store['default']=pd_s_n
store.close()
print('time used so far: %.2f seconds.'%(time.time()-ts))

pd_s_mid=pd.DataFrame(smidraw, index=feature_names)
print('check pd_s_mid')
#sys.exit()
fnh5=fntrunc+'.rawYBscore.inputtomid.h5'
print('writing to file ', fnh5)
store = pd.HDFStore(fnh5)
store['default']=pd_s_mid
store.close()
print('time used so far: %.2f seconds.'%(time.time()-ts))


pd_s_mid_n=pd.DataFrame(smid, index=feature_names)#normalized
print('check pd_s_mid_n')
#sys.exit()
fnh5=fntrunc+'.YBscore.inputtomid.h5'
print('writing to file ', fnh5)
store = pd.HDFStore(fnh5)
store['default']=pd_s_mid_n
store.close()
print('time used so far: %.2f seconds.'%(time.time()-ts))

from matplotlib.backends.backend_pdf import PdfPages
fnpdf=fntrunc+'.YBscore_matrices.pdf'
print('plotting to %s'%fnpdf)
pdf = PdfPages(fnpdf)
plt.figure(figsize=(7, 7))
matplotlib.rc('xtick', labelsize=6) 
matplotlib.rc('ytick', labelsize=6) 

v=pd_s.values
plt.hist(v.reshape((np.prod(v.shape),)), bins=50)
plt.title('hist pd_s')
plt.savefig(pdf, format='pdf')     
plt.close()
v=pd_s_n.values
plt.hist(v.reshape((np.prod(v.shape),)), bins=50)
plt.title('hist pd_s_n')
plt.savefig(pdf, format='pdf')     
plt.close()
v=pd_s_mid.values
plt.hist(v.reshape((np.prod(v.shape),)), bins=50)
plt.title('hist pd_s_mid')
plt.savefig(pdf, format='pdf')     
plt.close()
v=pd_s_mid_n.values
plt.hist(v.reshape((np.prod(v.shape),)), bins=50)
plt.title('hist pd_s_mid_n')
plt.savefig(pdf, format='pdf')     
plt.close()
pdf.close() 



#filter
if np.all(pd_s.index==pd_s.columns)==False:#problematic
  print('error: pd_s rownames and colnames are not the same. exit.')
  sys.exit()
#end if

if None!=gl:
  if set(gl)<=set(pd_s.index):
    pd_s_s=pd_s.loc[gl, gl]#subsetting using gl
    pd_s_n_s=pd_s_n.loc[gl, gl]#subsetting using gl


    #fngl
    #'tcga_brca_cell2015_htseqcounts.barcode.primary_solid_tumor.sym.PMID23000897_female.her2.genelist_rcge100sdne0.txt'
    m=re.compile('\.(genelist_.+?)\.txt').search(fngl)
    #m.group(0)
    #'.genelist_rcge100sdne0.txt'
    
    filterstr=m.group(1)    
    #'genelist_rcge100sdne0'

    fnh5_s=fntrunc+'.rawYBscore.filtered_'+filterstr+'.h5'
    print('writing to file ', fnh5_s)
    store = pd.HDFStore(fnh5_s)
    store['default']=pd_s_s
    store.close()

    fnh5_s=fntrunc+'.YBscore.filtered_'+filterstr+'.h5'
    print('writing to file ', fnh5_s)
    store = pd.HDFStore(fnh5_s)
    store['default']=pd_s_n_s
    store.close()


  else:#error
    print('error: some names in gl are not found in pd_s.index')
    sys.exit()
  #end if
#end if
print('time used so far: %.2f seconds.'%(time.time()-ts))
print('Done. total time used: %f'%(time.time()-ts))
#return
#end of main()

#if __name__=='__main__': main()


