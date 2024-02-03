
from zipfile import ZipFile
import csv

from tqdm import tqdm as progress

data_path=r'c:/users/lg/downloads/compressed/pii-detection-removal-from-educational-data.zip'
sample_submission_path='sample_submission.csv'

# open the zip file
def submission_sample(sample_size=50):

  assert isinstance(sample_size, int)
  assert sample_size>1     # we want at least two samples

  with ZipFile(data_path, 'r') as zip_file:
       # check if the csv file exists in the zip archive
       if sample_submission_path in zip_file.namelist():
           # open the csv file in text mode with the correct encoding
           with zip_file.open(sample_submission_path, 'r') as csv_file:
               csv_text = csv_file.read().decode('utf-8')  # read and decode the file contents

               # create a csv reader with the decoded text
               csv_reader = csv.reader(csv_text.splitlines())

               # read and display the first five rows (samples)
               for i, row in progress(enumerate(csv_reader)):
                   if   i < sample_size:
                       print(row)

       else:
          print(f"the file '{sample_submission_path}' was not found in the zip archive.")

if __name__=='__main__':
    submission_sample(sample_size=5)
    # it output
    #  ['row_id', 'document', 'token', 'label']
    #  ['0', '7', '9', 'B-NAME_STUDENT']
    #  ['1', '7', '10', 'I-NAME_STUDENT']
    #  ['2', '7', '482', 'B-NAME_STUDENT']
    #  ['3', '7', '483', 'I-NAME_STUDENT'
