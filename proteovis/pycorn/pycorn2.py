from .pycorn import pc_uni6

def load_uni_zip(path):
  fdata = pc_uni6(path)
  fdata.load()
  fdata.xml_parse()
  fdata.clean_up()
  return fdata