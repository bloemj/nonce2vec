#USAGE: python3 tokeniser.py [file to process -- one sentence per line]

from polyglot.text import Text
import logging
import logging.config
import sys
import pycld2

def tokenize(raw_text):
    """Tokenize raw_text with polyglot."""
    output = []
    try:
        text = Text(raw_text, hint_language_code='la')
        text.language='la'
        for sent in text.sentences:
            tokens = [token.lower().strip() for token in sent.words]
            output.append(' '.join(tokens))
    except ValueError as err:
        logger.debug('Skipping empty text sequence')
    except pycld2.error as err:
        logger.debug('{}. Skipping sequence'.format(str(err)))
    return '\n'.join(output)

logger = logging.getLogger(__name__)
f=open(sys.argv[1],'r')
rawtext = f.read()
print(tokenize(rawtext))

