#USAGE: python3 alnum_latin.py [file to process]
#Removes all non-alphanumeric characters from a text file, following Unicode (i.e. special alphabetic characters do get preserved)

import logging
import logging.config
import sys
import re

logger = logging.getLogger(__name__)
f=open(sys.argv[1],'r')
for line in f.readlines():
    line = re.sub(r"[^\w\d]+", " ", line, flags=re.UNICODE)
    print(line)




