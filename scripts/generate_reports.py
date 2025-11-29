# simple wrapper to call run_all
import subprocess, sys, os
p = subprocess.run([sys.executable, 'scripts/run_all.py'])
if p.returncode != 0:
    print('Pipeline failed with code', p.returncode)
else:
    print('Pipeline completed successfully')
