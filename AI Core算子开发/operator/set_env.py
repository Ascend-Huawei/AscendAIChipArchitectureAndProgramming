from topi.cce import te_set_version
import subprocess

te_set_version("1.3.T21.B880")
subprocess.call("python reduction.py",shell=True)