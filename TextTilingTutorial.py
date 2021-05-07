# # Discourse Segmentation Tutorial

# Before beginning, make sure that NLTK, NumPy, and Matplotlib have been installed.
# Please see the tutorial (https://veverica-hrdzava.github.io/discoursetutorial/) for
# directions on how to do this as well as an in-depth walkthrough of all the below steps.

# # TextTiling

# For TextTiling we will need to import texttiling from NLTK and PyLab from Matplotlib.
# For WindowDiff, we will need to import segmentation from NLTK.

from nltk.tokenize import texttiling
from matplotlib import pylab
from nltk.metrics import segmentation


# Before running, text will have to be preprocessed to ensure it has paragraph breaks, or an error will result.
# The below function cleans up the text we will be using in this tutorial, removing excess newlines.

def preprocess(text):
    no_ex_lines = ''
    for i in range(len(text)):
        if i < len(text)-1:
            if text[i] == '\n' and ch1[i+1] == '\n':
                no_ex_lines += '@'
            elif ch1[i] == '\n' and ch1[i+1] != '\n':
                no_ex_lines += ' '
            else:
                no_ex_lines += ch1[i]
        else:
            no_ex_lines += ch1[i]

    preprocessed = ''
    for i in no_ex_lines:
        if i == '@':
            preprocessed += '\n\n'
        else:
            preprocessed += i
    return preprocessed


# Adjust the line below so that it directs to the path of the .txt file on your computer.
# We will be looking at just the first chapter of this book and will preprocess it using the above function.

book  = open('/histeng.txt')
file = book.read()
book.close()

ch1 = file[950:22157]

text = preprocess(ch1)


# Two TextTiling objects will be instantiated, one for each of the demo_mode parameter settings.
# We will print the output of the tokenize() function of the second instantiation, which will give us the segments of the document.

text_tile = texttiling.TextTilingTokenizer(w=30,k=5,demo_mode=True)
text_tile_chunk = texttiling.TextTilingTokenizer(w=30,k=5,demo_mode=False)

sections = text_tile_chunk.tokenize(text)

#Print sections
print('TextTiling book sections:')
for i in range(len(sections)):
    print(i+1,':',sections[i],'\n')


# We will then look at a graphic representation of the gap scores for the pseudosentence breaks by calling
# tokenize() on the first instantiation of TextTile.

#Code below adapted from NLTK Demo
print('TextTiling visualization:')
gap_scores, smoothed_g_scores, depth_scores, breaks = text_tile.tokenize(text)
pylab.xlabel("Sentence Gap index")
pylab.ylabel("Gap Scores")
pylab.plot(range(len(gap_scores)), gap_scores, label="Gap Scores")
pylab.plot(range(len(smoothed_g_scores)), smoothed_g_scores, label="Smoothed Gap scores")
pylab.plot(range(len(depth_scores)), depth_scores, label="Depth scores")
pylab.stem(range(len(breaks)), breaks,use_line_collection=True)
pylab.legend()
pylab.show()


# We will then print out a list of all pseudosentence boundaries, with a value of 0 or 1, corresponding to whether a topic boundary was assessed.
# We will also print a list of all instances of 1 with numbers that will correspond to the pseudosentences, which we will print below.

print('Pseudosentence boundaries:',breaks,'\n')
print('Section boundaries:')
pseudo_breaks = []
for i in range(len(breaks)):
    if breaks[i] == 1:
        pseudo_breaks.append(i+1)
print(pseudo_breaks,'\n')


# # WindowDiff

# Before starting with WindowDiff, we will print the pseudosentences, so that we are able to compare the determination of
# the TextTiling with human segmentation judgements.

seqs = text_tile_chunk._divide_to_tokensequences(text)

print('List of pseudosentences:')

count = 1
for i in seqs:
    print('Pseudosentence #:',count)
    sequence = ''
    for i in i.__dict__['wrdindex_list']:
        sequence += i[0]+' '
    print(sequence)
    count += 1
print('\n')

# We will then need both the pseudosentence boundaries which we produced above (breaks) and a hypothetical human segmentation to compare.

text_tiling_segs = breaks
human_segs = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


# We will need to convert these lists to strings to use WindowDiff.

tt_string = ''
h_string = ''
for i in text_tiling_segs:
    tt_string += str(i)
for i in human_segs:
    h_string += str(i)
print('Boundaries as strings:')
print('TextTiling boundaries:',tt_string)
print('Hypothetical comparison boundaries:',h_string)
print('\n')

# Once these are have been converted, WindowDiff can be run using just one line of code.

success = segmentation.windowdiff(tt_string,h_string,3)
print('WindowDiff score:',success)
