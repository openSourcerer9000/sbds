"""Top-level package for Santa's Building Detection System."""

__author__ = """Sean Micek"""
__email__ = 'seanrm100@gmail.com'
__version__ = '0.1.0'

#drill down to the goods no matter where you're importing from
# if __package__ is None or __package__ == '':
try:
    # uses current directory visibility
    # from bldg_sam import *
    from remotesensing import *
except:
    # uses current package visibility
    # from .bldg_sam import *
    from .remotesensing import *