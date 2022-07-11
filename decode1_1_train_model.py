#!/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import time
import sys
import os
import re
import glob
import h5py
import pandas as pd

import sklearn.metrics
import scipy.stats
import scipy.stats.mstats_basic                                            

b_gpu=False
hostname=os.environ.get('HOSTNAME', '')
print('hostname is %s'%hostname)
hnparts=set(hostname.split('.'))
#if 'franklin' in hnparts or 'loge' in hnparts or 'orchestra' in hnparts: b_gpu=False
if False:
  pass
else:
  #scriptdir = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/..')
  #scriptdir='/home/hl136/zhangcheng/wd_nn/scripts'#on orchestra, harvard
  scriptdir='/home/cheng/wd_autoencoder'
  sys.path.append(scriptdir)#for path to lib_decode.py

  #os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32,force_device=true'
  #os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, dnn.enabled=True, scan.allow_output_prealloc=False, scan.allow_gc=True, optimizer_excluding=more_mem'
  #os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, dnn.enabled='+str(b_gpu)+', lib.cnmem=0.75, scan.allow_output_prealloc=False, scan.allow_gc=True, optimizer_excluding=more_mem, optimizer=fast_compile, exception_verbosity=high'
  #os.environ['THEANO_FLAGS'] = 'device=gpu, floatX=float32, dnn.enabled=True, scan.allow_output_prealloc=False, scan.allow_gc=True, optimizer_excluding=more_mem, optimizer=fast_compile, exception_verbosity=high, optimizer_including=cudnn'
  os.environ['THEANO_FLAGS'] = 'device=cuda, floatX=float32, dnn.enabled=True, scan.allow_output_prealloc=False, scan.allow_gc=True, optimizer_excluding=more_mem, optimizer=fast_compile, exception_verbosity=high, optimizer_including=cudnn'



  cr2=os.environ.get('CUDADIR','')
  if cr2!='': os.environ['CUDA_ROOT']=cr2
  cr=os.environ.get('CUDA_ROOT','')
  if cr=='':#error
    print('$CUDA_ROOT not set.')
    sys.exit()
  #end if
#end if 
import lib_decode

#np.random.seed(1337)  # for reproducibility


global X_train
global Y_train
global X_val#validation
global Y_val

global X_test
global Y_test

global gbl_epochs_to_save
global gbl_report_metrics
import keras.callbacks
n_epoch_to_report=50 # need to report training loss (without dropout) since dropout is activated when calculating loss during training.
#n_epoch_to_save_model_earlyepochs=100#when epoch < 1000
n_epoch_to_save_model_earlyepochs=10000#when epoch < 1000
n_epoch_to_save_model=10000# when epoch >=1000


y01=None
val_y01=None
X_train=None
Y_train=None
X_val=None
Y_val=None
X_test=None
Y_test=None
#ypred_bybatch=None
#gbl_ypred=None

def get_activations_keras111(model, layer, X_batch):#keras 1.1.1, https://github.com/fchollet/keras/issues/41#issuecomment-219262860
  from keras import backend as K
  get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
  activations = get_activations([X_batch,0])
  return activations

  
def get_activations_batchbybatch_keras111_ae(model, layer, X, batchsize=10):
  od=model.layers[layer].output_shape
  od=list(od)
  [nsample, ncol]=X.shape
  od[0]=nsample
  ret=np.zeros(shape=od)
  bs=batchsize#batchsize
  for i in range( int(np.ceil(1.0*nsample/bs)) ):
    act_i=get_activations_keras111(model, layer, X[i*bs:i*bs+bs,:])
    [d1,d2]=act_i.shape#in case the last portion of the sample doesnot have bs samples
    ret[i*bs:i*bs+d1,:]=act_i
    #print('i=',i, 'i*bs=',i*bs, act_i.shape)
  #end for i
  return ret


class my_callback_reg_ae(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    #print('in call back epoch = ', epoch) 
    # with current version of keras, epoch will increase AFTER this call back function returns, 
    # which means by the end of 1st epoch, the epoch variable here is set to 0
    # so in order to record correctly training performance at select epochs,
    # we need to use epoch+1
    if (0!=epoch+1 and epoch+1+gbl_epoch_trained>=1000 and (epoch+1+gbl_epoch_trained) % n_epoch_to_save_model ==0 ) or \
       (0!=epoch+1 and epoch+1+gbl_epoch_trained<1000 and  (epoch+1+gbl_epoch_trained) % n_epoch_to_save_model_earlyepochs ==0 ) or \
       (epoch+1 in gbl_epochs_to_save)  :
      #save model weights
      print('saving model...')
      [fn_ms, fn_mw]=gbl_cmi.save_model_and_weights(self.model, epoch+1+gbl_epoch_trained)
      print('model structure saved to file %s'%fn_ms)
      print('model weights saved to file %s'%fn_mw)
    #end if
    #print(gbl_report_metrics, n_epoch_to_report,  epoch+1, (epoch+1) % n_epoch_to_report)
    if gbl_report_metrics==False or n_epoch_to_report==0 or (epoch+1) % n_epoch_to_report !=0: return
    ts=time.time()
    print('\ntraining performance after epoch', epoch+1)
    print('calculating model predictions...')
    #ypred=lib_decode.get_activations_batchbybatch(self.model, len(self.model.layers)-1, X_train)#for plain nn, works with 1d input
    #ypred=get_activations_batchbybatch_keras111_ae(self.model, len(self.model.layers)-1, X_train)
    ypred=self.model.predict(X_train)
    #global gbl_ypred
    #gbl_ypred=ypred
    #print('stop to examine gbl_ypred');raw_input()
    #In [3]: gbl_ypred.shape
    #Out[3]: (99, 11840)
    #In [4]: X_train.shape
    #Out[4]: (99, 11840)
    #In [5]: Y_train.shape  
    #Out[5]: (99, 11840)
    #so, ypred has the same dimension with Y_train (and Y_train should be set to X_train)
    #print('ypred: {}, {}'.format(ypred.dtype, ypred.shape))
    #print('Y_train: {}, {}'.format(Y_train.dtype, Y_train.shape))

    ypr=np.reshape(ypred, -1)#to 1-D array
    ytr=np.reshape(Y_train, -1)#to 1-D array
    #print('ypr: {}, {}'.format(ypr.dtype, ypr.shape))
    #print('ytr: {}, {}'.format(ytr.dtype, ytr.shape))

    print('time used for calculating predictions: %f'%(time.time()-ts))
    #lib_decode.output_performance(y01, ypred01)
    #print('shapes:')
    #print(Y_train.shape)
    #print(ypred.shape)
    #ypred=np.squeeze(ypred)
    #print(ypred.shape)
    #mse=sklearn.metrics.mean_squared_error(Y_train, ypred)
    mse=sklearn.metrics.mean_squared_error(ytr, ypr)
    print('at epoch %d training mse=%f'%(epoch+1, mse))#actual mse without regularization

    r, pv_r=scipy.stats.pearsonr(ytr, ypr)
    print('at epoch %d training r=%f'%(epoch+1, r))#actual mse without regularization

    print('using np.mean and np.square mse=%f'%(np.mean(np.square(ytr-ypr))))

 
    #the test (validation) performance
    #if X_val!=None and Y_val!=None:
    if X_val is not None and Y_val is not None:
      ypredval=self.model.predict(X_val)
      ypr=np.reshape(ypredval, -1)#to 1-D array
      ytr=np.reshape(Y_val, -1)#to 1-D array
      mse=sklearn.metrics.mean_squared_error(ytr, ypr)
      print('at epoch %d validation mse=%f'%(epoch+1, mse))#actual mse without regularization
      r, pv_r=scipy.stats.pearsonr(ytr, ypr)
      print('at epoch %d validation r=%f'%(epoch+1, r))#actual mse without regularization

    #end if

    if X_test is not None and Y_test is not None:
      ypredval=self.model.predict(X_test)
      ypr=np.reshape(ypredval, -1)#to 1-D array
      ytr=np.reshape(Y_test, -1)#to 1-D array
      mse=sklearn.metrics.mean_squared_error(ytr, ypr)
      print('at epoch %d holdout test mse=%f'%(epoch+1, mse))#actual mse without regularization
      r, pv_r=scipy.stats.pearsonr(ytr, ypr)
      print('at epoch %d holdout test r=%f'%(epoch+1, r))#actual mse without regularization
      reteval_test=self.model.evaluate(X_test, Y_test)
      print('at epoch %d model eval on holdout test:   %f'%(epoch+1, reteval_test))#actual mse without regularization

    #end if


    reteval_train=self.model.evaluate(X_train, Y_train)
    reteval_val=self.model.evaluate(X_val, Y_val)
    #print('\nmodel eval on training,   {}\n'.format(self.model.evaluate(X_train, Y_train)))
    #print('\nmodel eval on valudation, {}\n'.format(self.model.evaluate(X_val, Y_val)))
    print('at epoch %d model eval on training:   %f'%(epoch+1, reteval_train))#actual mse without regularization
    print('at epoch %d model eval on valudation: %f'%(epoch+1, reteval_val))#actual mse without regularization



    ## r, pv_r=scipy.stats.pearsonr(ytr, ypr)
    ## print('training r=%f, pv_r=%f'%(r, pv_r))

    ## #rho, pv_rho=scipy.stats.spearmanr(Y_train, ypred)#TODO: calculating spearman rho cause error in HMS cluster. need to find out why.
    ## rho, pv_rho=scipy.stats.mstats_basic.spearmanr(ytr, ypr)#temporarily solved by using mstats_basic.spearmanr
    ## if type(pv_rho)!=float: pv_rho=pv_rho.tolist()
    ## print('training rho=%f, pv_rho=%f'%(rho, pv_rho))

    #this is autoencoder, so X_train is X_val, no need to calculate test metrics
    #In [22]: np.array_equal(X_train, X_val)
    #Out[22]: True

    #print('test performance after epoch', epoch+1)
    #print('calculating model predictions...')
    ##ypred=lib_decode.get_activations_batchbybatch(self.model, len(self.model.layers)-1, X_val)
    #ypred=get_activations_batchbybatch_keras111_ae(self.model, len(self.model.layers)-1, X_val)
    #ypred=np.squeeze(ypred)
    #print('time used for calculating predictions: %f'%(time.time()-ts))
    ##lib_decode.output_performance(val_y01, ypred01)
    #mse=sklearn.metrics.mean_squared_error(Y_val, ypred)
    ##mse=my_mse(Y_val, ypred)
    #print('test mse=%f'%mse)
    #r, pv_r=scipy.stats.pearsonr(Y_val, ypred)
    #print('test r=%f, pv_r=%f'%(r, pv_r))
    #rho, pv_rho=scipy.stats.mstats_basic.spearmanr(Y_val, ypred)#TODO: same.
    #pv_rho=pv_rho.tolist()
    #print('test rho=%f, pv_rho=%f'%(rho, pv_rho))

    print('time used for on_epoch_end: %f'%(time.time()-ts))
    return
  #end def on_epoch_end 
#end class



#my own constraint function, placed in keras/constrains
#/data2/syspharm/opt/lib/python2.7/site-packages/keras/constraints.py
from keras import backend as K
from keras.constraints import Constraint
class Zero(Constraint):  #added by zc@20161114
    '''Constrain the weights to be zero. to be used only with bias.
    '''
    def __call__(self, p):
        p *= K.cast(K.equal(p, 0.), K.floatx())
        return p

#def main():
print('works with keras 1.1.1 or 2.0.4 using model apis.')
print('input layer dropout is input corruption as in denoising autoencoder.')
b_logtofile=True
#b_logtofile=False

#fncfg='tempcfg.cfg'
#cmi=cnn_model_info(fncfg); cmi.output()
#cmi=cnn_model_info(); cmi.write_to_file(fncfg)
#sys.exit()
#cmi=cnn_model_info(); print(cmi.generate_summary_string()); sys.exit()

nargs=len(sys.argv)
if nargs!=2 and nargs!=3:
  print('usage: '+os.path.basename(__file__)+' <configuration file>')
  print('or   : '+os.path.basename(__file__)+' <template configuration file> --buildnew')
  sys.exit()
#end if
fpcfg=os.path.abspath(sys.argv[1])#'tempcfg.cfg'
cmi=lib_decode.cnn_model_info(fpcfg)
gbl_cmi=cmi
print('dump cmi.d')
for k in sorted(cmi.d.keys()): print(k, '=', cmi.d[k])


rs=1337
if 'random_seed' in cmi.d:
  rs=int(cmi.d['random_seed'])
#end if
#print('using random seed %d'%rs)
#np.random.seed(rs)

wd=os.path.dirname(fpcfg)
print('changing cwd to folder %s'%wd)
os.chdir(wd)
#cmi=cnn_model_info()
#cmi.write_to_file(fpcfg)

if nargs==3:
  if sys.argv[2]!='--buildnew':#
    print('unrecognized option: %s'%sys.argv[2])
    sys.exit()
  #end if
  cmi.d['timestamp']=time.strftime('%Y%m%d%H%M%S')
  fnnew=cmi.generate_config_filename()
  if fnnew != os.path.basename(fpcfg):
    fpcfg=os.path.join(wd, fnnew)
    cmi.write_to_file(fpcfg)
    print('config file renamed to ', fpcfg)
  #end if
  sys.exit()
#end if


orig_flags = os.environ.get('THEANO_FLAGS', '')
ompthrd=int(cmi.d['numthread'])
if ompthrd<1: ompthrd=1

if not b_gpu:
  os.environ['THEANO_FLAGS'] = 'openmp=true,exception_verbosity=high'
  os.environ['OMP_NUM_THREADS']=str(ompthrd)
  #print(cmi.d)
#end if not b_gpu
#sys.exit()  
ts=time.time()

#################################################################
#only import theano and keras after setting the environment variables
print('keras: using functional api to build model instead of graph')
print('only works with keras version 2 and no PReLU support yet.')

import keras
keras_version=1
if keras.__version__.startswith('1.'): keras_version=1
if keras.__version__.startswith('2.'): keras_version=2
print('keras_version=%d'%keras_version)

from keras.models import Model
from keras.layers import Input, Dense, merge 
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD 
if keras_version==2:
  from keras.utils.vis_utils import plot_model as kerasplot
elif keras_version==1:
  from keras.utils.visualize_util import plot as kerasplot#this is my modified version on the source
#need these in path:    progs = {'dot': '', 'twopi': '', 'neato': '', 'circo': '', 'fdp': '', 'sfdp': ''}
#added the following lines to ~/.bash_profile
#graphviz_path=/opt/graphviz-2.38.0/bin/
#PATH=~/installed/bin:$LSF_BINDIR:$LSF_SERVERDIR:$_CUR_PATH_ENV:$graphviz_path


#from keras.utils import np_utils
#end import 


#the following code for Corrupt needs to put into file 
#  /data2/syspharm/opt/lib/python2.7/site-packages/keras/layers/core.py
#######################################################################################
#adapted from code of dropout from keras theano_backend.py
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def corrupt(x, level, noise_shape=None, seed=None):
  """Sets entries in `x` to zero at random,
  while NOT scaling the entire tensor.
  # Arguments
      x: tensor
      level: fraction of the entries in the tensor
          that will be set to 0.
      noise_shape: shape for randomly generated keep/drop flags,
          must be broadcastable to the shape of `x`
      seed: random seed to ensure determinism.
  """
  if level < 0. or level >= 1:
    raise ValueError('Dropout level must be in interval [0, 1].')
  if seed is None:
    seed = np.random.randint(1, 10e6)
  if isinstance(noise_shape, list):
    noise_shape = tuple(noise_shape)

  rng = RandomStreams(seed=seed)
  retain_prob = 1. - level

  if noise_shape is None:
    random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
  else:
    random_tensor = rng.binomial(noise_shape, p=retain_prob, dtype=x.dtype)
    random_tensor = T.patternbroadcast(random_tensor,
                                         [dim == 1 for dim in noise_shape])
  x *= random_tensor
  #x /= retain_prob
  return x
#end def corrupt(x, level, noise_shape=None, seed=None):

#adapted from Dropout layer from keras core.py
from keras.engine import Layer 
from keras import backend as K
class Corrupt(Layer):
  """Applies corruption to the input.
  Corrupt consists in randomly setting
  a fraction `rate` of input units to 0 at each update during training time,
  which helps prevent overfitting.

  # Arguments
      rate: float between 0 and 1. Fraction of the input units to set to 0.
      noise_shape: 1D integer tensor representing the shape of the
          binary dropout mask that will be multiplied with the input.
          For instance, if your inputs have shape
          `(batch_size, timesteps, features)` and
          you want the dropout mask to be the same for all timesteps,
          you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.

  # References
      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
  """
  # @interfaces.legacy_dropout_support
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Corrupt, self).__init__(**kwargs)
    self.rate = min(1., max(0., rate))
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, _):
    return self.noise_shape

  def call(self, inputs, training=None):
    if 0. < self.rate < 1.:
      noise_shape = self._get_noise_shape(inputs)

      def corrupted_inputs():
        return corrupt(inputs, self.rate, noise_shape,
                           seed=self.seed)
      return K.in_train_phase(corrupted_inputs, inputs,
                              training=training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate}
    base_config = super(Corrupt, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
#end def class Corrupt(Layer):
#####################################################################

######################################
#from https://github.com/fzalkow/keras/blob/74c0af2010c2b2f2354bade0c0cc650fd2b0ae5d/keras/callbacks.py
import copy
import warnings

def get_actual_val_mse(model):
  ypred=model.predict(X_val)
  ypr=np.reshape(ypred, -1)#to 1-D array
  ytr=np.reshape(Y_val, -1)#to 1-D array
  print(['X_val.shape=', X_val.shape])
  print(['ypred.shape=', ypred.shape])
  print(['ypr.shape  =', ypr.shape])
  print(['ytr.shape  =', ytr.shape])
  mse=sklearn.metrics.mean_squared_error(ytr, ypr)
  return mse

class mycb_EarlyStopping(keras.callbacks.Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss', monitor_func=None, min_epoch=0,
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(mycb_EarlyStopping, self).__init__()
        if monitor_func==None: 
          print('monitor_func is None.')
          sys.exit()
        self.monitor_func=monitor_func
        print('monitor_func set to %s'%self.monitor_func)
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.min_epoch = min_epoch #minimum epoch to train before stopping
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_model_epoch = -1
        self.best_model_weights = copy.deepcopy(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        print('\nin on_epoch_end of cb_es: epoch=%d, self.min_epoch=%d'%(epoch, self.min_epoch))

        if epoch < self.min_epoch:
            return #
        print('self.best_model_epoch=%d'%self.best_model_epoch)
        current = self.monitor_func(model)
        print('current=%f'%current)
        #current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_model_epoch = epoch
            self.best_model_weights = copy.deepcopy(self.model.get_weights())
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_model_weights)

        #if self.stopped_epoch > 0 and self.verbose > 0:
        print('Epoch %05d: early stopping. Using weights from epoch %d according to lowest validation error, best=%f' % (self.stopped_epoch, self.best_model_epoch, self.best))
#end class
#####################################################################################

if not b_gpu:
  print('working with %d thread(s).'%ompthrd)
#end if not b_gpu
#sys.exit()

hostname='unknown'
if 'HOSTNAME' in os.environ: hostname=os.environ['HOSTNAME']
if '.' in hostname: hostname=hostname.split('.')[0]

#logfile=os.path.join(fpin, logfile)
#logfile=cmi.generate_log_filename()
#use a logfile that has the same main name as the input cfg file.
if fpcfg.endswith('.cfg'): 
  logfile=os.path.basename(fpcfg[:-4]+'.log')
else:#
  print('error: cannot generate log file name for config file %s. exit.'%fpcfg)
  sys.exit(1)
#end if
print('console output redirected to file %s'%logfile)
if b_logtofile:
  old_stdout=sys.stdout
  sys.stdout=open(logfile, 'ab', 0)
  sys.stderr=sys.stdout
#end if
lib_decode.dump_env()
print('config file: %s'%fpcfg)
timestr=time.strftime('%Y%m%d%H%M%S')# the start time
print('job started at: %s'%timestr)
print('using random seed %d'%rs)
np.random.seed(rs)


if not b_gpu:
  print('working on host %s with %d thread(s).'%(hostname, ompthrd))
#end if not b_gpu
#fnin='../dataset_ccle_2d_resize_100by100.h5'
fn_train=cmi.d['train_data_file']
fn_validate=cmi.d['validate_data_file']
fn_test=cmi.d['test_data_file']

if cmi.d.has_key('model_type') and cmi.d['model_type']=='autoencoder':
  pass
else:
  print('the config file should specify model_type as autoencoder')
  sys.exit()
#end if

print('loading training samples from ', fn_train)
print('before loading: %f M'%lib_decode.memory_usage_resource())


def load_data_from_file_list(fn_train, samples_to_excl=None):
  #load ';' separated file list
  m=None
  df=None
#  print(fn_train)
#  print(fn_train.split(';'))
  for fn_t in fn_train.split(';'):
    if fn_t.endswith('.gz') or fn_t.endswith('.txt') : pass
    else:
#      print(fn_t)
      print('training data should be tab-delimited text files.')
      sys.exit()
    df_t=pd.read_csv(fn_t, sep='\t', header=0, index_col=0, compression='infer')
    m_t=df_t.T.as_matrix()#nrow samples, ncol features
    print('m_t.shape:');print(m_t.shape)

    if m is None: m=m_t
    else: m=np.vstack( (m, m_t) )
    print('m.shape:');print(m.shape)

    if df is None: df=df_t
    else: df=pd.concat( [df, df_t] , axis=1)
    print('df.shape:');print(df.shape)
  #end for 

  #exclusion if present
  if samples_to_excl is None: 
    pass
  else:
#    df=df[:,np.isin(np.arange(df.shape[1]), samples_to_excl, invert=True)]

    df.drop(labels=df.columns[samples_to_excl], axis=1, inplace=True)
  #end if
  return df
#end def
  
# if ';' not in fn_train and ( fn_train.endswith('.gz') or fn_train.endswith('.txt') ):
#   pass
# else:
#   print('training data should be single tab-delimited text file.')
#   sys.exit()
# #end if
# df=pd.read_csv(fn_train, sep='\t', header=0, index_col=0, compression='infer')
# #df.index[1:3]
# #Index([u'AKT3', u'MED6'], dtype='object', name=u'gene_symbol')
# #df.columns[1:3]
# #Index([u'SW1116_LARGE_INTESTINE', u'NCIH1694_LUNG'], dtype='object')
# #df.shape
# #(18140, 1036)
# m=df.T.as_matrix()#nrow samples, ncol features
#m.shape
#(1036, 18140)
#end if  


les=list()
if cmi.d.has_key('epochs_to_save'):
  v=cmi.d['epochs_to_save']
  if v is None or v=='':
    pass
  vs=v.split(',')
  for vsi in vs:
    les.append(int(vsi))
  #end for vsi
#end if
print('les=', end=''); print(les)
gbl_epochs_to_save=les

gbl_report_metrics=True

if cmi.d.has_key('report_metrics'):
  v=cmi.d['report_metrics']
  if v is None or v=='':
    pass
  else:
    vs=int(v)
    if vs==0: gbl_report_metrics=False

#  print('report_metrics=', end=''); print(gbl_report_metrics)
#end if
print('report_metrics=', end=''); print(gbl_report_metrics)

l_se=list()
if cmi.d.has_key('training_samples_to_exclude'):
  v=cmi.d['training_samples_to_exclude']
  if v is None or v=='':
    pass
  vs=v.split(',')
  for vsi in vs:
    l_se.append(int(vsi))
  #end for vsi
#end if
print('l_se=', end='');print(l_se)
if len(l_se)==0: l_se=None
df=load_data_from_file_list(fn_train, l_se)
m=df.T.as_matrix()#nrow samples, ncol features

print('m.shape:');print(m.shape)
print('data loaded. time used: %.2f seconds.'%(time.time()-ts))
X=m
y=m
x_train=m
val_X=None
val_y=None
if fn_validate==fn_train or fn_validate=='' or fn_validate is None:
  val_X=x_train
  val_y=x_train
else:
  pass
  print('loading validation samples from ', fn_validate)

  #df=pd.read_csv(fn_validate, sep='\t', header=0, index_col=0, compression='infer')
  #m=df.T.as_matrix()#nrow samples, ncol features
  df=load_data_from_file_list(fn_validate)
  m=df.T.as_matrix()#nrow samples, ncol features


  #m=load_data_from_file_list(fn_validate)
  print('data loaded. time used: %.2f seconds.'%(time.time()-ts))
  val_X=m
  val_y=m
#end
#independept holdout test data
X_test=None
Y_test=None
print('fn_test=', fn_test)

# if fn_test is None: print('None')
# if fn_test=='': print('blank')
# sys.exit()

if fn_test is None or fn_test=='':
  pass
else:
  df=load_data_from_file_list(fn_test)
  m=df.T.as_matrix()#nrow samples, ncol features
  X_test=m.astype(np.float32)
  Y_test=m.astype(np.float32)
#end if
#sys.exit()
#end if
X_train=X
Y_train=y


X_val=val_X
Y_val=val_y


X_train=X_train.astype(np.float32)
Y_train=Y_train.astype(np.float32)

X_val=  X_val.astype(np.float32)
Y_val=  Y_val.astype(np.float32)

#sys.exit()#debug
#nb_pool = int(cmi.d['poolsize'])
weight_init=cmi.d['weight_init']
batch_size = int(cmi.d['batch_size'])#1000#128
n_epoch_to_train = int(cmi.d['epoch_to_train'])#100#12

l_nndense=[]
if cmi.d['nndense']!='': l_nndense=[int(v) for v in cmi.d['nndense'].split('-')]
n_dense=len(l_nndense)
print('number dense layers: %d'%n_dense)
print(l_nndense)
loss_func=cmi.d['loss_func']
optmr=cmi.d['optimizer']
optmr2=optmr
if cmi.d.has_key('optimizer2'): optmr2=cmi.d['optimizer']

actfun_default='linear'
actfun=actfun_default
act_l1=None#activity regularizer
if cmi.d.has_key('act_l1'): act_l1=float(cmi.d['act_l1'])
if act_l1==0: act_l1=None

if cmi.d.has_key('activation'): actfun=cmi.d['activation'].lower()
if actfun=='': actfun=actfun_default
print('actfun set to %s'%actfun)
if actfun=='prelu':
  from keras.layers.advanced_activations import PReLU 

lr=float(cmi.d['lr'])
print('epoch to train: %d'%n_epoch_to_train)
print('loss function: %s'%loss_func)
print('optimizer: %s with lr=%f'%(optmr, lr))
print('optimizer2: %s with lr=%f'%(optmr2, lr))



# input_dropout_rate=None
# if cmi.d.has_key('input_dropout_rate'):
#   input_dropout_rate=float(cmi.d['input_dropout_rate'])
#   print("input_dropout_rate=%f"%input_dropout_rate)
# if input_dropout_rate<=0 or input_dropout_rate>=1:
#   input_dropout_rate=None
#   print('dropout omitted.')
# #sys.exit()

input_corrupt_rate=None
if cmi.d.has_key('input_corrupt_rate'):
  input_corrupt_rate=float(cmi.d['input_corrupt_rate'])
  print("input_corrupt_rate=%f"%input_corrupt_rate)
if input_corrupt_rate<=0 or input_corrupt_rate>=1:
  input_corrupt_rate=None
  print('corrupt omitted.')
#sys.exit()
input_gaussian_noise_stddev=None
if cmi.d.has_key('input_gaussian_noise_stddev'):
  input_gaussian_noise_stddev=float(cmi.d['input_gaussian_noise_stddev'])
  print("input_gaussian_noise_stddev=%f"%input_gaussian_noise_stddev)
if input_gaussian_noise_stddev<=0:
  input_gaussian_noise_stddev=None
  print('gaussian noise omitted.')
#sys.exit()

input_gaussian_dropout_rate=None
if cmi.d.has_key('input_gaussian_dropout_rate'):
  input_gaussian_dropout_rate=float(cmi.d['input_gaussian_dropout_rate'])
  print("input_gaussian_dropout_rate=%f"%input_gaussian_dropout_rate)
if input_gaussian_dropout_rate<=0:
  input_gaussian_dropout_rate=None
  print('gaussian dropout omitted.')
#sys.exit()



#rebuild model each time and just load previous weights
#  because Graph model cannot be properly loaded (duplicate output node error)

model=None
fnmw=None
epoch_trained=0
fnmw,  epoch_trained=cmi.try_find_weights() #do not look for files. save time.
gbl_epoch_trained=epoch_trained
print('before bulding model: %f M'%lib_decode.memory_usage_resource())
if b_gpu==True:
  #from theano.sandbox.cuda import dnn_version as version
  from theano.gpuarray.dnn import version# as version

#  print('cudnn version:', version())
p_verbose=2#one line per epoch
if model==None:
  print('building model...')
  fun_optmr=optmr
  if optmr=='sgd':
    fun_optmr=keras.optimizers.sgd(lr=lr)
    print('with optimizer sgd, lr=%f'%(lr))
  elif optmr=='rmsprop':
    fun_optmr=keras.optimizers.RMSprop(lr=0.010, rho=0.9, epsilon=1e-6)
  #end if

  fun_optmr2=optmr2
  if optmr2=='sgd':
    fun_optmr2=keras.optimizers.sgd(lr=lr)
    print('with optimizer sgd, lr=%f'%(lr))
  elif optmr2=='rmsprop':
    fun_optmr2=keras.optimizers.RMSprop(lr=0.010, rho=0.9, epsilon=1e-6)
  #end if


  #now build an autoencoder model
  #code taken from https://blog.keras.io/building-autoencoders-in-keras.html
  
  from keras.layers import Input, Dense
  from keras.models import Model, Sequential
  if keras_version==1:
    from keras.regularizers import l2, activity_l2, l1l2, l1  
  if keras_version==2: 
    from keras.regularizers import l2, l1  
    from keras.regularizers import l1_l2 as l1l2  


  #from keras.constraints import Zero
  bcnst=None
  print('b_constraint initialized to None')
  if cmi.d.has_key('bconstraint0'):
    val=cmi.d['bconstraint0']
    if val=='1':
      bcnst=Zero()
      print('bcnst set to Zero()')

  breg=lambda: None
  print('b_reg initialized to None')
  if cmi.d.has_key('breg'):
    val=cmi.d['breg']
    if val=='l1l2': 
      breg=l1l2
      print('b_reg set to l1l2')
      if bcnst!=None:
        bcnst=None
        print('reset bcnst to None.')
  #end if

  print('all activation functions set to ', actfun)
  actreg=lambda: None
  if act_l1!=None: actreg=lambda: l1(act_l1)
  print('hidden layer activity regularizer set to ', actreg())

  nsamp_in=m.shape[0]
  nfeat_in=m.shape[1]

  #encoding_dim = l_nndense[0]
  #print('single hidden layer autoencoder. encoding_dim=%d'%encoding_dim)  

  inputlyr = Input(shape=(nfeat_in,))# this is our input placeholder

  ###################################################################
  #sdae, according to vincent chapter 3.5:
  #1. layer-wise training: 
  #  for each layer, train it weights on     * corrupted *   input;
  #  after training, apply this layer on the * uncorrupted * input to produce input for the next layer.
  #2. training on stacked:
  #  once all the layers are trained, stack these layer to form the final model. Fine tune the model on corrupted/uncorrupted (not specified in the paper) input.

  ###################################################################

  #the following code works with keras 2 only
  input_dim_i=nfeat_in
  x_train_i=x_train
  i=0
  l_dae=list()
  l_hidden=list()
  
   
  # l_nndense has the form of [3000, 50, 3000] because the input layer dimension is already specified by the data. We need to append the output dimension to complete the model.
  l_nndense.append(nfeat_in)
  for i, nn_i in enumerate(l_nndense, start=1): # number of neurons for layer i
    print('building dae for layer %d'%i)
    input_i = Input(shape=(input_dim_i,))# 
    corrupt_i=None
    if input_gaussian_dropout_rate!=None:
      from keras.layers.noise import GaussianDropout
      corrupt_i= GaussianDropout(rate=input_gaussian_dropout_rate)(input_i)
      print('gaussian noise with stddev %f for input layer.'%input_gaussian_dropout_rate)
    elif input_gaussian_noise_stddev!=None:
      from keras.layers.noise import GaussianNoise
      corrupt_i= GaussianNoise(stddev=input_gaussian_noise_stddev)(input_i)
      print('gaussian noise with stddev %f for input layer.'%input_gaussian_noise_stddev)
    elif input_corrupt_rate!=None:
      corrupt_i= Corrupt(rate=input_corrupt_rate)(input_i)
      print('corrupt with rate %f for input layer.'%input_corrupt_rate)
    #end if
    if corrupt_i==None:#no corruption
      corrupt_i=input_i#use input directly
    hidden_i=None
    if actfun=='prelu':
      hidden_i = PReLU()(Dense(nn_i, input_dim=input_dim_i, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg())(corrupt_i))
    else:
      hidden_i=Dense(nn_i, activation=actfun, input_dim=input_dim_i, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg())(corrupt_i)
    output_i=None
    if actfun=='prelu':
      output_i=PReLU()(Dense(input_dim_i, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg())(hidden_i))
    else:
      output_i=Dense(input_dim_i, activation=actfun, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg())(hidden_i)
    dae_i=Model(inputs=input_i, outputs=output_i)
    dae_i_enc=Model(inputs=input_i, outputs=hidden_i)

    print('compiling model for dae_i ...') 
    dae_i.compile(optimizer=fun_optmr, loss=loss_func)#
    if fnmw==None: #if model weights exist in file fnmw, the weights will be loaded so no need to train here.
      if len(l_nndense)>1:#only pretrain each hidden layer when there's more than 1 hidden layer
        print('pretraining dae_i ...') 

        dae_i.fit(x_train_i, x_train_i,
                      epochs=100,#n_epoch_to_train,
                      batch_size=batch_size, verbose=p_verbose,
                      shuffle=True)#, callbacks=l_cb, validation_data=(X_val, X_val))


    #after training
    l_dae.append(dae_i)
    l_hidden.append(hidden_i)
    pred_i=dae_i_enc.predict(x_train_i)


    input_dim_i=nn_i 
    x_train_i=pred_i
  #end for nn_i

  #now after training a list of daes, stack them together
  sdae=Sequential()
  #sdae.add(inputlyr)
  #for dae in l_dae: sdae.add(dae)
  #for h_i in l_hidden: sdae.add(h_i)
  print('check l_dae')
#  raw_input()
  for dae in l_dae:
    for il in range(len(dae.layers)):
#         <keras.layers.core.Dense at 0x7f330eea9d90>,
#         <keras.layers.advanced_activations.PReLU at 0x7f330eea9cd0>,
#         <keras.layers.core.Dense at 0x7f330f0199d0>,
#         <keras.layers.advanced_activations.PReLU at 0x7f330eea9ed0>]
      if '.Dense' in str(dae.layers[il]):
        print('found 1st dense layer at il=%d'%il)
        sdae.add(dae.layers[il]) #add the 1st dense layer from dae
        #check to see if there's any advanced_activations layer after this
        if il+1 < len(dae.layers) and '.advanced_activations' in str(dae.layers[il+1]):
          sdae.add(dae.layers[il+1]) #
          print('with the advanced activation layer after it.')
        break
    #end for il
  #end for dae


  model=sdae
  model.compile(optimizer=fun_optmr2, loss=loss_func)#

  ##  #encoded = Dense(encoding_dim, activation=actfun)(inputlyr)# using relu results in decoder layer w all 0 and middle layer activation all 0 
  ##  # dropout1=None
  ##  # if input_dropout_rate!=None:
  ##  #   if keras_version==1:
  ##  #     dropout1= Dropout(p=1/input_dropout_rate)(inputlyr)
  ##  #   elif keras_version==2:
  ##  #     dropout1= Dropout(rate=input_dropout_rate)(inputlyr)
  ##  # #end if
  ##  # lyr1=inputlyr
  ##  # if dropout1!=None: lyr1=dropout1


  ##  corrupt1=None
  ##  if input_gaussian_dropout_rate!=None:
  ##    from keras.layers.noise import GaussianDropout
  ##    corrupt1= GaussianDropout(rate=input_gaussian_dropout_rate)(inputlyr)
  ##    print('gaussian noise with stddev %f for input layer.'%input_gaussian_dropout_rate)
  ##  elif input_gaussian_noise_stddev!=None:
  ##    from keras.layers.noise import GaussianNoise
  ##    corrupt1= GaussianNoise(stddev=input_gaussian_noise_stddev)(inputlyr)
  ##    print('gaussian noise with stddev %f for input layer.'%input_gaussian_noise_stddev)
  ##  elif input_corrupt_rate!=None:
  ##    corrupt1= Corrupt(rate=input_corrupt_rate)(inputlyr)
  ##    print('corrupt with rate %f for input layer.'%input_corrupt_rate)
  ##  #end if

  ##  lyr1=inputlyr
  ##  if corrupt1!=None: lyr1=corrupt1

  ##  # this is the size of our encoded representations

  ##  #encoded=None
  ##  #if keras_version==1: 
  ##  #  encoded = Dense(encoding_dim, activation=actfun, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg)(lyr1)
  ##  #elif keras_version==2:
  ##  #  if actfun=='prelu':
  ##  #    encoded = PReLU()( Dense(encoding_dim, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg(), activity_regularizer=actreg)(lyr1) )
  ##  #  else:
  ##  #    encoded = Dense(encoding_dim, activation=actfun, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg(), activity_regularizer=actreg)(lyr1)

  ##  #decoded=None
  ##  #if keras_version==1:
  ##  #  decoded = Dense(nfeat_in, activation=actfun, b_constraint=bcnst, b_regularizer=breg())(encoded)# "decoded" is the lossy reconstruction of the input
  ##  #elif keras_version==2:
  ##  #  if actfun=='prelu':
  ##  #    decoded = PReLU()( Dense(nfeat_in, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg())(encoded) )# "decoded" is the lossy reconstruction of the input
  ##  #  else:
  ##  #    decoded = Dense(nfeat_in, activation=actfun, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg())(encoded)# "decoded" is the lossy reconstruction of the input


  ##  currentlyr=lyr1
  ##  # l_nndense has the form of [3000, 50, 3000] because the input layer dimension is already specified by the data. We need to append the output dimension to complete the model.
  ##  l_nndense.append(nfeat_in) 
  ##  for nn_i in l_nndense: # number of neurons for layer i
  ##    newlyr=None
  ##    if keras_version==1: 
  ##      newlyr = Dense(nn_i, activation=actfun, b_constraint=bcnst, b_regularizer=breg(), activity_regularizer=actreg())(currentlyr)
  ##    elif keras_version==2:
  ##      if actfun=='prelu':
  ##        newlyr = PReLU()( Dense(nn_i, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg(), activity_regularizer=actreg())(currentlyr) )
  ##      else:
  ##        newlyr = Dense(nn_i, activation=actfun, kernel_initializer=weight_init, bias_constraint=bcnst, bias_regularizer=breg(), activity_regularizer=actreg())(currentlyr)
  ##    #end if 
  ##    currentlyr=newlyr
  ##  #end for nn_i
  ##  decoded=currentlyr
  ##   
  ##  autoencoder =None
  ##  if keras_version==1: autoencoder = Model(input=inputlyr, output=decoded)# this model maps an input to its reconstruction
  ##  elif keras_version==2: autoencoder = Model(inputs=inputlyr, outputs=decoded)# this model maps an input to its reconstruction

  ##  
  ##  ##use 2 separate models, encoder and decoder, to track the internal states of this autoencoder
  ##  #encoder = Model(input=inputlyr, output=encoded)# this model maps an input to its encoded representation
  ##  #encoded_input = Input(shape=(encoding_dim,))# create a placeholder for an encoded (32-dimensional) input
  ##  #decoder_layer = autoencoder.layers[-1]# retrieve the last layer of the autoencoder model
  ##  #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))# create the decoder model
  ##  print('compiling model...') 
  ##  autoencoder.compile(optimizer=fun_optmr, loss=loss_func)#
  ##  model=autoencoder



  ############no plotting to save time.
  fnmodelplot=fpcfg+'.png'
  print('plotting model structure to file %s'%fnmodelplot)
  if keras_version==1: kerasplot(model, to_file=fnmodelplot, recursive=True)#this is with keras 1.1 
  elif keras_version==2: kerasplot(model, to_file=fnmodelplot, show_shapes=True)
  print('model compiled. time used so far: ', time.time()-ts, ' seconds.')
  print('after building model: %f M'%lib_decode.memory_usage_resource())

  if fnmw!=None:
    print('loading weights from file %s'%fnmw)
    model.load_weights(fnmw)
#end if

  
print('before model.fit')
#sys.exit()
#model.fit(x=X_train, y=Y_train , batch_size=batch_size, nb_epoch=n_epoch_to_train, verbose=2, validation_data=(val_X, val_y), callbacks=[cb])
cb=my_callback_reg_ae()
l_cb=[cb]

b_es=False
if cmi.d.has_key('early_stop') and cmi.d['early_stop']=='1' : b_es=True

if b_es:
  es_patience=int(np.ceil(n_epoch_to_train * 0.6))
  print('early stopping -- patience set to %d'%es_patience)
  #cb_es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=es_patience, verbose=0, mode='min') 
  cb_es=mycb_EarlyStopping(monitor='val_loss', min_delta=0, patience=es_patience, verbose=1, 
                         min_epoch=int(np.ceil(es_patience/3.0)),
                         mode='min', monitor_func=get_actual_val_mse) 
  l_cb.append(cb_es)
#end if b_es

if keras_version==1:
  model.fit(x_train, x_train,
                    nb_epoch=n_epoch_to_train,
                    batch_size=batch_size,
                    shuffle=True, callbacks=l_cb,
                    validation_data=(X_val, X_val))#, callbacks=[cb]) #no validation_data, no callback to speed up the training
elif keras_version==2:
  model.fit(x_train, x_train,
                    epochs=n_epoch_to_train,
                    batch_size=batch_size,
                    shuffle=True, callbacks=l_cb,verbose=p_verbose,
                    validation_data=(X_val, X_val))

print('after model.fit: %f M'%lib_decode.memory_usage_resource())

epoch_trained_current=n_epoch_to_train

if b_es:
  print('early stopping -- stopped_epoch = %d'%cb_es.stopped_epoch)
  print('early stopping -- best_model_epoch = %d'%cb_es.best_model_epoch)
  epoch_trained_current=cb_es.best_model_epoch
#end if
#epoch_trained_current=cb_es.stopped_epoch
#save model
print('saving model...')
[fn_ms, fn_mw]=cmi.save_model_and_weights(model, epoch_trained+epoch_trained_current)
print('model structure saved to file %s'%fn_ms)
print('model weights saved to file %s'%fn_mw)

timeused=time.time()-ts
print('Time used: %.2f seconds.'%timeused)
timestr=time.strftime('%Y%m%d%H%M%S')# the start time
print('job finished at: %s'%timestr)
  
if b_logtofile:
  sys.stdout.close()
#  return
  
#end of main()
#if __name__ == '__main__':
#  main()

