import sys

import warnings

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
