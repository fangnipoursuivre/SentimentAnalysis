"""
Created on May 1st, 2020 - München

Sentiment Analysis of Health Authority Feedback

@author: Dr. FANG Ni
"""

# PDF Extraction: Conclusion Section of CHMP Assessment Report for Xarelto (page:52-56)

import re
from datetime import datetime
from PyPDF2 import PdfFileReader

# ------------------------------------------------------------------#
pdffile = 'WC500057122.pdf'                 # pdf source file       #
textfile = 'WC500057122.txt'                # extracted text file   #
finalfile = 'WC500057122_final.txt'         # finalised text file   #
# ------------------------------------------------------------------#

with open(pdffile, "rb") as abc:
    pdf = PdfFileReader(abc)
    info = pdf.getDocumentInfo()
    pages = pdf.getNumPages()

    print('\n')
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'), '%s has been opened.' % pdffile)
    print('\n')
    print('Document information is summarized as follows.')
    print('\n')
    print (info)
    print ('number of pages: %d' % pages)

    # extract conclusion section: page 52-56

    def extraction(textfile, m, n): # m: starting page (actual page-1), n: total number of interest pages from the starting page
       # create an empty text file
       text_file = open(textfile, 'w+')

       for x in range(n):
           text_file = open(textfile,'a+')
           page = pdf.getPage(m+x)

           content = [page.extractText()]  # extract page content

           for sentence in content:
               split_content = re.split("(?<=[.!?])\s+", sentence)
               cleaned_content = '\n'.join(split_content)
               text_file.write(cleaned_content)


    def finalise_text(textfile, finalfile): # finalise text to fit one-sentence one-line format
        with open(textfile) as infile, open(finalfile, 'w') as outfile:
            words = ['EMEA 2008'] # page title needs to be removed
            for line in infile:
                if not any(word in line for word in words):
                    if not line.strip():
                        continue # skip the empty line
                    outfile.write(line)

    extraction(textfile,51,5)
    finalise_text(textfile, finalfile)

    print('\n')
    print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'), 'Conclusion section of %s has been extracted and saved to the local directory with the file name as %s.' % (pdffile, finalfile))