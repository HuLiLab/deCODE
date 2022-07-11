#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import time
import sys
import os
import re
import glob
#np.random.seed(1337)  # for reproducibility
gi_default_random_seed=1337
#
#model saving and loading are implemented with json and h5,
#  model structure saved in json format, while the weights saved in h5.
#  In this way the models will be saved as device independent compared to the cPickle method, which is device dependent.
#  (i.e. model build on GPU cannot be loaded for CPU.)

import h5py


class cnn_model_info:
  l_fields=['timestamp',
            'random_seed',
            'numthread',
            'poolsize',
            'conv_layers',
            'nndense',
            'weight_init',
            'n_class',
            'regression',
            'model_type',
            'loss_func',
            'optimizer',
            'lr',
            'batch_size',
            'epoch_to_train',
            'train_data_file',
            'validate_data_file',
            'test_data_file',
            'model_file',
            'log_file']

  #          'epoch_trained'
  #these fields are generated after 1st run
  l_fields_opt=['regression', 'model_type', 'activation','optimizer2', 
                'input_gaussian_noise_stddev',
                'input_gaussian_dropout_rate',
                'input_corrupt_rate', 'input_dropout_rate', 
                'training_samples_to_exclude', # comma seperated list of 0-based numbers, for training samples to exclude. Can be empty
                'report_metrics',# 0 or 1, if 0, don't calculate mse and R in callback
                'epochs_to_save', # comma seperated list of 1-based numbers, at these epochs the model weight will be saved. can be empty
                'epoch_to_pretrain', # default is hard-coded to 10000 in the training script; can be empty
                'breg',#l1l2 otherwise not
                'bconstraint0',#1 otherwise not
                'act_l1', 'early_stop']# early_stop=<0|1>
  
  def __init__(self, config_file=None):
    self.config_file=config_file
    #
    self.d=dict()
    if config_file!=None: self.d=self.load_from_file(config_file)
    else:#
      self.init_as_sample()
    #end if
    #check
    missing_fields=sorted(list(set(self.l_fields)-set(self.d.keys())))
    optional_fields=set(self.l_fields_opt)
    if len(missing_fields)==0 \
      or set(missing_fields) <= optional_fields:
      # or (len(missing_fields)==1 and missing_fields[0]=='regression'):
      pass
    else:
      print('the following fields are missing in config file:')
      print(missing_fields)
      sys.exit()
    #end if
    #check lr format
    lr=float(self.d['lr'])
    lr_c2=float(self.str_add_dot(self.str_rm_dot(self.d['lr'])))
    if abs(lr-lr_c2)<1e-9:#passed
      pass 
    else:#
      print('error: invalid learning rate lr=%s'%self.d['lr'])
      sys.exit()
    #end if
    
  #end def __init__
  def init_as_sample(self):
    self.d=dict()
    self.d['timestamp']=time.strftime('%Y%m%d%H%M%S')
    self.d['batch_size']='10'
    self.d['random_seed']='1337'

    self.d['epoch_to_train']='10'
    self.d['poolsize']='2'
    self.d['numthread']='32'

    self.d['conv_layers']='25-3-3'
    self.d['nndense']='128' #can have the form of 1024-128 for multiple dense layers

    self.d['weight_init']='he_normal'
    self.d['n_class']='2'
    self.d['regression']='0'#0 or 1 only. if has key regression and value==1, this is a regression model

    self.d['loss_func']='categorical_crossentropy'
    self.d['optimizer']='sgd'#
    self.d['lr']='0.01'

    self.d['train_data_file']='dataset_ccle_2d_resize_200by200.h5'
    self.d['validate_data_file']='dataset_ccle_2d_resize_200by200.h5'
    self.d['test_data_file']=''

    self.d['model_file']=''
    self.d['log_file']=''
    #self.d['epoch_trained']='0'
  #end def 
  def str_rm_dot(self, float_in_str):
    #make a float value suitable for file name by removing its decimal dot.
    #
    s=str(float_in_str)
    if '.' in s: s=s.replace('.','')
    return s
  #end def str_rm_dot(self, float_in_str):
  def str_add_dot(self, float_in_str):
    #if s starts with 0, it is considered a fraction.
    #  otherwise s remains the same as input
    s=str(float_in_str)
    if s.startswith('0'):
      s='0.'+s[1:]
    #end if
    return s
  #end def str_add_dot(self, float_in_str):

  def generate_summary_string(self):
    lo=['',
        'poolsize%d'%int(self.d['poolsize']),
        'nndense%s'%('-'.join(self.d['nndense'].split('-'))),
        'convlayer%s'%('-'.join(self.d['conv_layers'].split('-'))),
        'optimizer%s'%self.d['optimizer'], 
        self.str_rm_dot('lr%s'%self.d['lr']),
        'rs%s'%self.d['random_seed']
       ]
    if self.d.has_key('regression') and self.d['regression']=='1':#regression
      lo[0]='reg'
    elif self.d.has_key('model_type') and self.d['model_type']=='autoencoder':#regression
      lo[0]='autoencoder' 
    else:
      lo[0]='class%d'%int(self.d['n_class'])
    #end if
    res='_'.join(lo)
    return res
  def generate_config_filename(self):
    ss=self.generate_summary_string()
    res='config_'+ss+'_'+self.d['timestamp']+'.cfg'
    return res
  #end def generate_config_filename(self):

  def generate_model_weight_filename_wildcard(self):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_weight_epoch*.h5'
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_weight_epoch*.h5')
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if

    return res
  #end def generate_model_weight_filename_wildcard(self):

  def generate_model_weight_filename(self, epoch_trained):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_weight_epoch%d.h5'%epoch_trained
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_weight_epoch%d.h5'%epoch_trained)
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if
    #print(res)
    return res
  #end def generate_model_weight_filename(self, epoch_trained):

  def generate_model_structure_filename(self):
    #ss=self.generate_summary_string()
    #res='config_'+ss+'_'+self.d['timestamp']+'.model_structure.json'
    res=None
    if self.config_file.endswith('.cfg'): 
      res=os.path.basename(self.config_file[:-4]+'.model_structure.json')
    else:#
      print('error: cannot generate log file name for config file %s. exit.'%self.config_file)
      sys.exit(1)
    #end if

    return res
  #end def generate_model_structure_filename(self):

  def generate_log_filename(self):
    ss=self.generate_summary_string()
    res='config_'+ss+'_'+self.d['timestamp']+'.log'
    return res
  #end def generate_log_filename(self):
  def enum_weights(self):
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
    l_mw_fn=glob.glob(mw_fn_wc)
    l_epoch=[0]*len(l_mw_fn)
    fn_mw=None
    if len(l_mw_fn)!=0:#found model weights, select one of them
      print('found existing model weights')
      re_e=re.compile('epoch(\d+)')
      maxe=-1; maxp=None
      for i in range(len(l_mw_fn)):
        mdl=l_mw_fn[i]
        ms=re_e.findall(mdl)
        if ms==None or len(ms)!=1:#error
          print('model file name cannot be parsed: %s'%mdl)
          sys.exit()
        #end if
        #print(ms)
        e=int(ms[0])
        l_epoch[i]=e
        if e>maxe: maxe=e; maxp=i
        print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
      #end for i
      fn_mw=l_mw_fn[maxp]
      epoch_trained=maxe
    else:
      print('nothing found using wildcard ', mw_fn_wc,'\n')
    #end if
    os.chdir(old_wd)
    return [l_mw_fn, l_epoch]
  #end def enum_weights(self):

  def try_find_weights(self):#just find the files for model weights without loading it
    #because Graph model structures cannot be loaded (duplicate output node error)
    #  so build the model each time and load the weights
    from keras.models import model_from_json
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
    l_mw_fn=glob.glob(mw_fn_wc)
    l_epoch=[0]*len(l_mw_fn)
    fn_mw=None
    if len(l_mw_fn)!=0:#found model weights, select one of them
      print('found existing model weights')
      re_e=re.compile('epoch(\d+)')
      maxe=-1; maxp=None
      for i in range(len(l_mw_fn)):
        mdl=l_mw_fn[i]
        ms=re_e.findall(mdl)
        if ms==None or len(ms)!=1:#error
          print('model file name cannot be parsed: %s'%mdl)
          sys.exit()
        #end if
        #print(ms)
        e=int(ms[0])
        l_epoch[i]=e
        if e>maxe: maxe=e; maxp=i
        print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
      #end for i
      fn_mw=l_mw_fn[maxp]
      epoch_trained=maxe
    #end if
    os.chdir(old_wd)
    return [fn_mw, epoch_trained]
  #end def try_find_weights(self):

  def try_load_model_with_weights(self):
    from keras.models import model_from_json
    #search for previously trained models
    wd=os.path.dirname(os.path.abspath(self.config_file))
    old_wd=os.getcwd()
    os.chdir(wd)

    model=None
    epoch_trained=0

    fn_ms=self.generate_model_structure_filename()#file of model structure
    if os.path.exists(fn_ms):
      #model structure found, now try to find model weights
      mw_fn_wc=self.generate_model_weight_filename_wildcard()#model weight
      l_mw_fn=glob.glob(mw_fn_wc)
      l_epoch=[0]*len(l_mw_fn)
      fn_mw=None
      if len(l_mw_fn)!=0:#found model weights, select one of them
        print('found existing model weights')
        re_e=re.compile('epoch(\d+)')
        maxe=-1; maxp=None
        for i in range(len(l_mw_fn)):
          mdl=l_mw_fn[i]
          ms=re_e.findall(mdl)
          if ms==None or len(ms)!=1:#error
            print('model file name cannot be parsed: %s'%mdl)
            sys.exit()
          #end if
          #print(ms)
          e=int(ms[0])
          l_epoch[i]=e
          if e>maxe: maxe=e; maxp=i
          print('%dth model file %s with %d epochs trained.'%(i+1, mdl, l_epoch[i]))
        #end for i
        fn_mw=l_mw_fn[maxp]
        epoch_trained=maxe
      #end if
      if fn_mw!=None:
        print('loading latest model weights from %s trained for %d epochs...'%(fn_mw, epoch_trained))
     
        model=model_from_json(open(fn_ms, 'rb').read())
        model.load_weights(fn_mw)
      #end if
    else:#no model structure
      pass
    #end if
    os.chdir(old_wd)
    return [model, epoch_trained]
  #end def try_load_model_with_weights(self):

  def save_model_and_weights(self, model, epoch_trained):
    fn_ms=self.generate_model_structure_filename()
    fn_mw=self.generate_model_weight_filename(epoch_trained)
   
    json_string = model.to_json()
    open(fn_ms, 'wb').write(json_string)
    model.save_weights(fn_mw, overwrite=True)
    return [fn_ms, fn_mw]
  #end def save

  def output(self):
    for k in self.l_fields+self.l_fields_opt:
      print(k, self.d[k])
  #end def output
  def load_from_file(self, config_file):
    s_allfields=set(self.l_fields+self.l_fields_opt)
    
    #print('in load_from_file:', s_allfields)
    re_eq=re.compile('^([_a-zA-Z0-9]+)=([,${} ;_a-zA-Z0-9./-]*)$')
    d=dict()
    for l in open(config_file, 'r'):
      sl=l.strip()
      if sl.startswith('#') or sl=='': continue#comments or empty line
      m=re_eq.match(sl)
      #print(sl)
      #print(m.groups())
      if m==None:
        print('error: line in config file not recognized.')
        print(sl)
        sys.exit()
      #print(m.group(1), m.group(2))
      k=m.group(1)
      v=m.group(2)

      if k in s_allfields: d[k]=v
      else:
        print('warning: unrecognized key discarded - %s'%k)
    #end for
    if 'random_seed' not in d:
      d['random_seed']=str(gi_default_random_seed)
    return d
  #end def load_from_file
  def write_to_file(self, config_file):
    print('writing config to file %s'%config_file)
    fout=open(config_file, 'wb')
    allfields=self.l_fields+self.l_fields_opt
    #print(allfields)
    #print(self.d.keys())
    for k in allfields:
      if k in self.d:
        fout.write('%s=%s\n'%(k, self.d[k]))
    #end for k
    fout.close()
    return
  #end def write_to_file

#end class cnn_model_info:
def memory_usage_resource():
  import resource
  import sys
  rusage_denom = 1024.
  if sys.platform == 'darwin':
    # ... it seems that in OSX the output is different units ...
    rusage_denom = rusage_denom * rusage_denom
  mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
  return mem
#end def memory_usage_resource():
def load_data_w_channel(fnh5):#
  f=h5py.File(fnh5, 'r')
  g=f['default']
  N_alloc=len(g.keys())

  #get size first
  dm=g[g.keys()[0]].value
  #actually this is to flip the order of all axes
  #for 2D, just transpose
  #for 3D, need to swap the 1st and the 3rd axes.
  #  not figured out for more than 3 dims yet.

  #dmt=dm.transpose()
  dmt=np.swapaxes(dm, 0, 2)

  [nchannel, nr, nc]=dmt.shape

  print( 'allocating space: %d samples, %d channels, %d by %d'%(N_alloc, nchannel, nr, nc))
  X=np.ndarray(shape=(N_alloc, nchannel, nr, nc))#allocate a large space first
  y=np.ndarray(shape=(N_alloc))
  i=0

  for k in g.keys():
    if i>=N_alloc:#
      print( 'more data than N_alloc. exit.');sys.exit()
    #end if
    data_map3d=g[k].value

    act=None

    spltk=k.split('-') 
    if len(spltk)==3:
      comp, sample, act=spltk
    elif len(spltk)>3:
      sample=spltk[-2]
      act=spltk[-1]
      comp='-'.join(spltk[:-2])
    else:#error
      print('error: cannot parse k ',k)
      sys.exit(1)
    #end if
    act=float(act)
    #print k, act, data_map3d.shape, data_map3d.transpose().shape;raw_input()
    if act==None:#error
      print('error: act==None for k=',k)
      sys.exit()
    #end if
    #dmt=data_map3d.transpose()
    dmt=np.swapaxes(data_map3d, 0, 2)
    #dmt=dmt[:crop_size,:crop_size]
    X[i]=dmt
    y[i]=act
    i+=1
  #end for k
  print('Total %d samples loaded.'%i)
  print('data shape:', X.shape)
  return [X,y]
#end def
def to_categorical(y, nb_classes=None):
  '''Convert class vector (integers from 0 to nb_classes)
  to binary class matrix, for use with categorical_crossentropy
  '''
  y = np.asarray(y, dtype='int32')
  y = np.asarray(y, dtype=np.int16)

  if not nb_classes:
      nb_classes = np.max(y)+1
  Y = np.zeros((len(y), nb_classes))
  for i in range(len(y)):
      Y[i, y[i]] = 1.
  return Y

def load_and_format_datalist_guess_cutoff_and_channel(fnl_sc_sep, cutoff_ccle=8, cutoff_gsk=5000, cutoff_def=None):#fnl_sc_sep: semi-colon separated file list.
  #cutoff_ccle=8
  #cutoff_gsk=5000
  print('cutoff_ccle=%d, cutoff_gsk=%d'%(cutoff_ccle, cutoff_gsk))
  
  N_CLASS=2
  [X, y, l_title]=[None, None, None]
  lf=list()
  fnl_sc_sep=fnl_sc_sep.strip('; ')
  if ';' in fnl_sc_sep: lf=[v.strip() for v in fnl_sc_sep.split(';')]
  else: lf.append(fnl_sc_sep.strip())
  print('there are %d files in list %s.'%(len(lf), fnl_sc_sep))

  n_files=len(lf)
  #get the size of each input file to use for space allocation
  N_alloc=0
  for fn in lf:
    f=h5py.File(fn, 'r')
    g=f['default']
    n_i=len(g.keys())
    N_alloc+=n_i
  #end for fn

  #get the internal dimension of the data matrix by looking at the 1st file in list
  fn1=lf[0]
  f=h5py.File(fn1, 'r')
  g=f['default']
  n1=len(g.keys())
  #get size first
  dm=g[g.keys()[0]].value
  sh=dm.shape
  nchannel=1
  nr=nc=None
  #the h5 file is generated with rhdf5 and the dimensions need rotation
  if len(sh)==2:#1 channel
    dmt=dm.transpose()
    [nr, nc]=dmt.shape
  elif len(sh)==3:#>1 channel
    dmt=np.swapaxes(dm, 0, 2)
    [nchannel, nr, nc]=dmt.shape
  else:#number of channels not supported
    print('error: number of dimensions %d not supported. exit.'%len(sh))
  #end if

  print( 'allocating space: %d samples, %d channels, %d by %d'%(N_alloc, nchannel, nr, nc))
  X=np.ndarray(shape=(N_alloc, nchannel, nr, nc), dtype=np.float32)#allocate a large space first
  #y=np.ndarray(shape=(N_alloc))
  y=np.ndarray(shape=(N_alloc,2), dtype=np.float32)
  l_title=['']*N_alloc
  i=0

  #for each file in list
  for fn in lf:
    fnh5=fn
    f=h5py.File(fnh5, 'r')
    g=f['default']
    istart=i
    ytemp=np.ndarray(shape=(len(g)))
    for k in g.keys():
      if i>=N_alloc:#
        print( 'more data than N_alloc. exit.');sys.exit()
      #end if

      dm=g[k].value
      if nchannel==1:
        dmt=dm.transpose()
        X[i][0]=dmt
      elif nchannel>1:
        dmt=np.swapaxes(dm, 0, 2)
        X[i]=dmt
      #end if
      l_title[i]=k
      act=None
      spltk=k.split('-') 
      if len(spltk)==3:
        comp, sample, act=spltk
      elif len(spltk)>3:
        sample=spltk[-2]
        act=spltk[-1]
        comp='-'.join(spltk[:-2])
      else:#error
        print('error: cannot parse k ',k)
        sys.exit(1)
      #end if
      act=float(act)
      #print k, act, dm.shape, dm.transpose().shape;raw_input()
      if act==None:#error
        print('error: act==None for k=',k)
        sys.exit()
      #end if

      ytemp[i-istart]=act
      i+=1
      if i%1000==0:
        print('memory usage after loading %d samples: %f M'%(i, memory_usage_resource() ))
    #end for k
    iend=i
    cutoff=None
    if cutoff_def!=None: cutoff=cutoff_def
    elif 'ccle' in os.path.basename(fn).lower(): cutoff=cutoff_ccle
    elif 'gsk' in os.path.basename(fn).lower(): cutoff=cutoff_gsk
    else:#not supported
      print('unrecognized dataset: %s. exit.'%fn)
      sys.exit(1)
    #end if
    print('sensitive/resistant cutoff set to %d'%cutoff)
    ytemp_cls=[0 if v<cutoff else 1 for v in ytemp]
    ytemp_cat=to_categorical(ytemp_cls, N_CLASS)#copied function definition
    y[istart:iend,]=ytemp_cat
  #end for fn
  #now return the actual used values
  return [X[:i,:,:,:], y[:i,:], l_title[:i]]
#end def load_and_format_datalist_guess_cutoff_and_channel(fnl_sc_sep):#semi-colon separated.
def get_activations(model, layer, X_batch):
  import theano
  get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
  activations = get_activations(X_batch) # same result as above
  return activations


def get_activations_batchbybatch(model, layer, X, batchsize=10):
  od=model.layers[layer].output_shape
  od=list(od)
  [nsample, nfilter, nrow, ncol]=X.shape
  od[0]=nsample
  ret=np.zeros(shape=od)
  bs=batchsize#batchsize
  for i in range( int(np.ceil(1.0*nsample/bs)) ):
    act_i=get_activations(model, layer, X[i*bs:i*bs+bs,:,:,:])
    [d1,d2]=act_i.shape#in case the last portion of the sample doesnot have bs samples
    ret[i*bs:i*bs+d1,:]=act_i
    #print('i=',i, 'i*bs=',i*bs, act_i.shape)
  #end for i
  return ret


def get_activations_batchbybatch_nn(model, layer, X, batchsize=10):#a plain neural network, 1st layer is a dense layer.
  od=model.layers[layer].output_shape
  od=list(od)
  #[nsample, nfilter, nrow, ncol]=X.shape
  [nsample, nfeature]=X.shape
  od[0]=nsample
  ret=np.zeros(shape=od)
  bs=batchsize#batchsize
  for i in range( int(np.ceil(1.0*nsample/bs)) ):
    act_i=get_activations(model, layer, X[i*bs:i*bs+bs,:])
    [d1,d2]=act_i.shape#in case the last portion of the sample doesnot have bs samples
    ret[i*bs:i*bs+d1,:]=act_i
    #print('i=',i, 'i*bs=',i*bs, act_i.shape)
  #end for i
  return ret


def output_performance(y01, ypred01, posval=0, negval=1):#0 as sensitive/positive and 1 as insensitive/negative
  n=len(y01)
  npred=len(ypred01)
  if n!=npred:
    print('n=%d, npred=%d, not equal. exit.'%(n, npred))
    sys.exit()
  #end if
  y01=np.asarray(y01)
  ypred01=np.asarray(ypred01) 
  tp=np.sum( np.logical_and(y01==posval, ypred01==posval) )
  tn=np.sum( np.logical_and(y01==negval, ypred01==negval) )

  fp=np.sum( np.logical_and(y01==negval, ypred01==posval) )
  fn=np.sum( np.logical_and(y01==posval, ypred01==negval) )

  precision=1.0*tp/(tp+fp)
  recall=1.0*tp/(tp+fn)
  true_negative_rate=1.0*tn/(tn+fp)
  accuracy=1.0*(tp+tn)/n
  
  f1=0.0
  if tp==0: f1=0.0#tp is 0 then both precision and recall are 0
  else:
    f1=2.0*precision*recall/(precision+recall)
  #end if
  npos=np.sum(y01==posval)
  nppos=np.sum(ypred01==posval)
  print('total %d samples, %d (%f) positive, %d (%f) predicted positive.'%(n, npos, 1.0*npos/n, nppos, 1.0*nppos/n))
  print('true positive= ', tp, 'false positive=', fp)
  print('true negative= ', tn, 'false negative=', fn)
  print('precision=', precision)
  print('recall=', recall)
  print('true_negative_rate=', true_negative_rate)
  print('accuracy=', accuracy)

  print('f1 score=', f1)
#end def



def calc_crossentropy(y, ypred):
  #y and ypred should of the same shape, nsample * nclass(2)
  y=np.asarray(y)
  ypred=np.asarray(ypred)

  sh1=y.shape
  sh2=ypred.shape
  if sh1!=sh2:
    print('error: (calc_crossentropy) shape mismatch: shape1=',sh1, 'shape2=',sh2)
    sys.exit()
  #end if
  if sh1[1]!=2:#error
    print('error: (calc_crossentropy) shape not of 2 classes: shape1=',sh1, 'shape2=',sh2)
    sys.exit()
  #end if

  ce=0.0
  #ce2=0.0
  n=sh1[0]
  for i in range(n):
    ce+= y[i,0]*np.log(ypred[i,0]) + (1-y[i,0])*np.log(1-ypred[i,0])
    #ce2+= y[i,1]*np.log(ypred[i,1]) + (1-y[i,1])*np.log(1-ypred[i,1])

  #
  ce/=-n
  #ce2/=-n
  #print('ce=',ce, ', ce2=', ce2)#ce and ce2 are the same
  return ce
#end def calc_crossentropy(y, ypred):



def dump_env():
  print('dumping environment variables:')
  for k in sorted(os.environ.keys()):
    print(k, os.environ[k])
#end def dump_env



def stratified_k_fold_class(y, k):
  rs=np.random.RandomState(None)
  y=np.asarray(y, dtype='int')
  #print y
  n=len(y)
  arr=np.zeros((n,3), dtype='int')
  arr[:,0]=y
  arr[:,1]=[i for i in range(n)]
  #print arr
  arr=arr[  y.argsort(), :]
  #print arr
  idx_shuffle=np.zeros((n,), dtype='int')

  for val in np.unique(y):
    #print 'val=', val 
    idx=arr[:,0]==val
    #print 'idx=', idx
    #print 'arr[idx,:]'
    #print arr[idx,:]
    selected_idx=np.nonzero(idx)[0]
    #print 'selected_idx=', selected_idx
    rs.shuffle(selected_idx)
    #print 'selected_idx=', selected_idx
    idx_shuffle[idx]=selected_idx
    #print 'idx_shuffle=', idx_shuffle
  #end for

  arr=arr[idx_shuffle, :]
  #print arr
  for i in range(n):
    arr[i,2]=i % k
  #end for i
  #print arr
  #now select each test fold
  ll=list()
  for i in range(k):
    idx_te=np.nonzero(arr[:,2]==i)[0]
    idx_tr=np.nonzero(arr[:,2]!=i)[0]
    #print idx_tr, idx_te
    #orig idx
    tr=arr[idx_tr,1]
    te=arr[idx_te,1]
    #print tr, te
    ll.append([tr, te])
  #
  #print 'result:'
  for tr, te in ll:
    #print tr, te
    pctr=1.0*np.sum(y[tr]==0)/len(tr) 
    pcte=1.0*np.sum(y[te]==0)/len(te)
    print(pctr, pcte)
  return ll
#def stratified_k_fold(y, k):


