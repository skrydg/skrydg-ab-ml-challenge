import pathlib

class Env:
  def __init__(self, input_directory, output_directory):
    self.input_directory = pathlib.Path(input_directory)
    self.output_directory = pathlib.Path(output_directory)