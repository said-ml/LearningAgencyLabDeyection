import sys
import tensorflow as tf

from random  import seed
from numpy.random import seed as np_seed

from typing import Optional
from typing import Union
from typing import Callable

import warnings


# define a simple decorator(inner function) to mesure the execution of a function
def measure_time(func: Callable) -> Callable:
    '''
    args: one argument, is the function that we want to decorated(e,g functions that:
    build the model, train the model, optimize the model, ...etc)
    '''

    # define the wrapper function that executes the original function(e,g arg func)
    def wrapper(*args, **kwargs) -> float:
        '''
        the wrapper function have the same argument as the object(e,g arg)function
        '''
        start_time = time()
        # call the object function to process and manipulate ...
        result = func(*args, **kwargs)
        end_time = time()

        processing_time = end_time - start_time
        print(f'the processing time is {processing_time:.2f} secondes')
        return result

    return wrapper


class Configurations:

      def __init__(self,
                   seed_arg:int=42,
                   accelerator:Optional[Union['CPU', 'GPU', 'TPU']]=None)->None:

          '''
          :param seed: to make the outputs stable
          :param accelator: threre are three options 'CPU' as automatic choice and other 'GPU' and 'TPU'
          are setting with intention
          '''
          self.seed_arg=seed_arg
          self.accelerator=accelerator

      def seed_all(self)->None:
          seed(self.seed_arg)
          np_seed(self.seed_arg)

      @measure_time
      def Accelerator(self, set_memory:bool=True)->str:

          gpus = tf.config.list_physical_devices('GPU')
          tpus = tf.config.list_physical_devices('TPU')

          if gpus:
            self.accelerator='GPU'
            return self.accelerator
            if set_memory:
              # Enabling growth memory for GPUs
              try:
                  for gpu in gpus:
                      tf.config.experimental.set_memory_growth(gpu, True)
                  print('memory growth enable for all GPUs')

              except RuntimeError as e:
                  print("Error enabling memory growth for GPUs:", e)
          elif tpus:

                  print('to enable "TPU" you must write down some more code')
                  # detect and init the TPU
                  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

                  # instantiate a distribution strategy
                  tf.tpu.experimental.initialize_tpu_system(tpu)
                  tpu_strategy = tf.distribute.TPUStrategy(tpu)

                  print('you need to add: with tpu_strategy.scope():before define your model')

                  self.accelerator = 'TPU'
                  return self.accelerator

          else:
              self.accelerator = 'CPU'
              return self.accelerator

      def manage_warnings(self)->None:
          if not sys.warnoptions:
             warnings.filterwarnings('ignore')



if __name__ == '__main__':
    CFG=Configurations()
    CFG.seed_all()
    acce=CFG.Accelerator(set_memory=True)
    print(acce)