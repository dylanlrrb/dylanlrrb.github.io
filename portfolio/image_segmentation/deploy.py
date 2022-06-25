import subprocess
import os
import sys
import boto3

model_name = sys.argv[1]
s3_parent_dir = sys.argv[2]

out = subprocess.run(['tensorflowjs_converter', '--input_format=keras', '--output_format=tfjs_graph_model', f'./models/{model_name}.h5', f'./tfjs_models/{model_name}'])
if out.returncode > 0:
  print(out)
print(f'The exit code for tensorflowjs_converter was: {out.returncode}')

if out.returncode == 0:
  print (f'uploading model to s3\nbuilt-model-repository/{s3_parent_dir}/{model_name}')
  directory = f'./tfjs_models/{model_name}'
  s3_client = boto3.client('s3')

  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
      filename = f.split('/')[-1]
      s3_client.upload_file(
        f,
        'built-model-repository',
        f'{s3_parent_dir}/{model_name.replace("/", "_")}/{filename}',
        ExtraArgs={'ACL': 'public-read'}
      )
      print('.', end=' ')
  print('Done')