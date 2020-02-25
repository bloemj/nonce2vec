#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import getopt
import codecs
import re
import itertools
import mmap
import random
import xml.etree.ElementTree as ET
from polyglot.text import Text

#This script can create a dataset of one-word Wikipedia article titles and their first sentence (the definitional sentence), when given a WikiExtractor dump as input.

def main(argv=None):

  try:
    opts, args = getopt.getopt(argv, "hi:o:l:k:f:c:", ["help", "indir=", "outfile=", "minsentlen=", "keytermsfile=", "freqdict=", "freqcutoff="])
  except getopt.GetoptError:
    getout()

  indir=""
  outfile=""
  keytermsfile=""
  minsentlength = 10
  freqdict=""
  freqcutoff = 50
  for opt, arg in opts:
    if opt in ("-h", "--help"):
      usage()
      sys.exit()
    if opt in ("-i", "--indir"):
      indir = arg
    if opt in ("-o", "--outfile"):
      outfile = arg
    if opt in ("-k", "--keytermsfile"): #Terms that should be forced into the evaluation set (if meeting the requirements)
      keytermsfile = arg
    if opt in ("-f", "--freqdict"): #Word frequency list for cutoff
      freqdict = arg
    if opt in ("-c", "--freqcutoff"): #Frequency cutoff for being an eligible instance of a term and definition
      freqcutoff = int(arg)
    if opt in ("-l", "--minsentlen"): #Minimal sentence length
      minsentlength = int(arg)

  if not (indir and outfile and freqdict):
    getout()
    
  if argv is None:
    argv = sys.argv

  sample(indir, outfile, keytermsfile, minsentlength, freqdict, freqcutoff)

def getout():
  usage()
  sys.exit(2)

def usage():
  print('sample-wiki-n2vevalset.py -i <inputdir> -o <outputfile> -f <frequency dictionary file> (-l <minimum sentence length> [default 10]) (-c <frequency cutoff> [default 50]) (-l <key terms file>) (-h)')
  
def tokenize(raw_text, lowercase):
    """Tokenize raw_text with polyglot."""
    output = []
    text = Text(raw_text, hint_language_code='la')
    text.language='la'
    for sent in text.sentences:
        if lowercase:
            tokens = [token.lower().strip() for token in sent.words]
            output.append(tokens)
        else:
            tokens = [token.strip() for token in sent.words]
            output.append(tokens)
    return output

def sample(indir, outfile, keytermsfile, minsentlength, freqdict, freqcutoff):

    keyterms = []
    if keytermsfile:
        with codecs.open(keytermsfile, encoding="utf-8") as k:
            for line in k:
                keyterms.append(line.lower().strip())

    freqs = {}
    with codecs.open(freqdict, encoding="utf-8") as q:
        for line in q:
            line = line.strip().split()
            if len(line) > 1:
                if int(line[0]) >= freqcutoff:
                    freqs[line[1].strip()] = line[0]

    candidatedefinitions = {}
    for filename in os.listdir(indir):
        print(filename)
        with codecs.open(indir + '/' + filename, encoding="utf-8") as f:
            for line in f:
                r = re.search('title=\"(.*)\"', line)
                if r:
                    title = r.group(1).strip().lower()
                    #title = re.sub(r'\(.*\)','',title).strip() #remove parentheses content after title
                else:
                    continue
                next(f)
                next(f)

                firstpar = next(f).strip()
                if not firstpar:
                    continue
                #firstpar = re.sub(r'\(.*\)','',firstpar).strip() #remove parentheses content in first sentence, which is often foreign language material
                candidatedefinition = tokenize(firstpar, 1)[0]
                i = 0
                candidatedefinition = ['___' if token==title else token for token in candidatedefinition]
                for token in candidatedefinition:
                    if len(token)>1:
                        i=i+1
                if i >= minsentlength:
                    candidatedefinition = ' '.join(candidatedefinition)
                    #pattern = re.compile('[\W\s]+')
                    candidatedefinition = re.sub(r"[^\w\d]+", " ", candidatedefinition, flags=re.UNICODE)
                    if '___' in candidatedefinition:
                        if not ' ' in title.strip():    #only single word title pages
                            if title.strip() in freqs:
                                if len(title.strip()) > 1:
                                    candidatedefinitions[title.strip()] = candidatedefinition
    
    definitiontest = []

    print(len(candidatedefinitions))
    definitionsample = random.sample(list(candidatedefinitions), 1000)
    definitiontrn = definitionsample[:700]
    definitiontest.extend(definitionsample[700+len(definitiontest):])

    for term in keyterms:
        if term in candidatedefinitions:
            print(term)
            if term not in definitiontest:
                definitiontest.append(term)
                definitiontest.pop(0)
    
    with codecs.open(outfile + '700.train', "w", encoding="utf-8") as trn:
        for key in definitiontrn:
            noncesentence = candidatedefinitions[key]
            trn.write(key + '\t' + noncesentence + '\n')
            
    with codecs.open(outfile + '300.test', "w", encoding="utf-8") as test:
        for key in definitiontest:
            noncesentence = candidatedefinitions[key]
            test.write(key + '\t' + noncesentence + '\n')
                


if __name__ == '__main__':
    main(sys.argv[1:])

