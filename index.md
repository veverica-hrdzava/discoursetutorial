# Discourse Segmentation Tutorial: TextTiling
In this short Discourse Parsing tutorial, we will be discussing how to use the `TextTiling` tool available in the Natural Language Toolkit, a simple tool for breaking text up into separate topic subsections. This will be followed by a discussion of WindowDiff, which can assess the performance of machine discourse segmentation and a brief discussion of discourse segmentation and parsing broadly, with some more recent applications.


## 1.1. Preliminaries: Introduction
The structure of language is often understood to be largely intrasentential in nature. In this understanding of structure, a sentence is constructed of a finite number of structural building blocks. The parts of a sentence express relationships: a subject, a verb, maybe a direct object or an indirect object or some other modifier, and each element is a member of a subset of a finite set of structural categories: overt subjects can be nouns, or pronouns, or sentences themselves, etc. There are many methods, developed within NLP and Linguistics that can parse a sentence into its constituent parts and offer insight into the nature of the relationships of these elements.

Perhaps less frequently discussed is how these different sentences connect and interact with each other in a document, and how connected sentences form subsections. We know language is not composed of a bunch of random sentences jumbled together. Instead, complex ideas are formed by stringing together multiple sentences. In elementary school, we learn how to write a three-point paragraph, where each of the five paragraphs has a clear purpose. One to introduce the focus of the  How might we be able to separate these separate paragraphs? How are the pieces in these paragraphs internally similar? And how are they different from surrounding paragraphs? How might we go about separating a longer document, which includes many paragraphs, with paragraphs that seem to be part of meaningful units of paragraphs?

TextTiling is an algorithm, and a tool available through the NLTK, for doing this. As an input it takes a string composed of paragraphs, and its output is a list of single or multi-paragraph groupings.

## 1.2. Preliminaries: Required Libraries
The code in this tutorial will be run on `Python 3`. You will also need to install `Natural Language Toolkit`, `NumPy`, and `Matplotlib`. In the repository, a [.py]() file, a [Jupyter Notebook .ipynb]() file, and the [text file]() we will be parsing, are available for download if you would like to follow along.

### Python
Python can be downloaded on [here](https://www.python.org/downloads/). Simply click `Download Python 3.9.5` (number subject to changes!) and follow the directions on the install.

### NLTK
Mac and Unix users can download the `Natural Language Toolkit` ([NLTK for short](https://www.nltk.org/)) using pip. [Here are detailed instructions on the NLTK install page for Windows, Mac, and Unix users](https://www.nltk.org/install.html). Run the line below in your terminal to install through pip (line adapted from NLTK install page).

```
pip install nltk
```

### NumPy
[`NumPy`](https://numpy.org/) can also be downloaded using pip, [here are instructions for installing NumPy](https://numpy.org/install/). If you can install through pip, use the command line below (line from NumPy install page).

```
pip install numpy
```

### Matplotlib
[`Matplotlib`](https://matplotlib.org/), like the others, can quickly be installed using Matplotlib. [Here's the page for Matplotlib installation](https://matplotlib.org/stable/users/installing.html). (Line below adapted from Matplotlib install page).

```
pip install matplotlib
```

Note that you may have to use the following code to ensure that it is installed in Python 3 and not Python 2.

```
pip3 install nltk
pip3 install numpy
pip3 install matplotlib
```

## 2. NLTK TextTiling Tool
`TextTiling` is a discourse segementer available as a part of the `Natural Language Toolkit`. The algorithm was developed by [Hearst (1997)](https://www.aclweb.org/anthology/J97-1003.pdf) as a way to segement text into multiparagraph units. The underlying logic is an intuitive one--sentences within topic subsections have more similarity with one another than they do with sentences in parts of a document that have a different topic. By moving across the document and comparing the words in one section of text with those sections before and after it, we can understand how

[NLTK documentation](https://www.nltk.org/api/nltk.tokenize.html) is available, and the source code can be found [here](https://www.nltk.org/_modules/nltk/tokenize/texttiling.html).

The algorithm preprocesses a document by levelling case and removing punctuation. It then tokenizes the preprocessed text, separates words into pseudosentences of *w* length and compares *k* pseudosentences before and after each pseudosentence boundaries. This is done by calculating a *depth score*, representing the similarity between the compared text before and after each break. Those boundaries that show the most dissimilarity between the words in *k* sentences on either side are marked as the location where a boundary is made. When graphically represented, these boundaries will form depth score *valleys* (to use the terminology employed by Hearst). See Hearst (1997: 50) for details on depth score calculation. These boundaries are then assigned to the closest paragraph break, giving an output of 'chunks' of paragraphs. This ensures that the segmentation of the text does not disturb paragraphs, and explains why paragraph breaks are necessary for any text input.

The first thing we will need to do will be to import our libraries. For this we will need to import `texttiling` and `pylab` for a visualization of the segemented text.

```python
from nltk.tokenize import texttiling
from matplotlib import pylab
```

Text may need to be preprocessed. For this tutorial, we will be using a book from [Project Gutenberg](https://www.gutenberg.org/), [Charles Dickens' *A Child's History of England*](https://www.gutenberg.org/ebooks/699). Each line in the this .txt file ends with `\n`, and while it's not strictly necessary to remove these, the text is cleaner when it's removed. Because the TextTiling algorithm splits on paragraphs, we must insure that, while these extra `\n` characters are removed, the paragraph breaks remain. Failure to have paragraph breaks will result in an error when the TextTiling code is run.

For the purpose of this, we will just be looking at the first chapter from the book. Once this has been loaded and preprocessed, we are ready to run the TextTiling algorithm and segment the topic.

```python
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

#Edit the below file to the path of the file, available [here]( ), on your computer.
book  = open('/histeng.txt')
file = book.read()
book.close()
ch1 = file[950:22157]
text = preprocess(ch1)
```

Once text has been loaded and preprocessed, we're ready to initialize the TextTilingTokenizer. There are a number of parameters, among them:
- `w`: pseudosentence length (default `20`)
- `k`: boundary comparison length (default `10`)
- `stopwords`: a custom list of stopwords to be filtered out (default `None`, NLTK English stopwords)
- `demo_mode`: `False` if a list of the text chunks is desired, `True` if a graphic representation output is desired

`w` and `k` can be calibrated to document genre, but Hearst found that the defaults of `w=20` and `k=10` worked well across different document types. See Hearst (1997: 54) for a discussion of these parameters.

We will initialize two objects, one for each of the `demo_mode` parameter settings.

```python
text_tile_chunk = nltk.TextTilingTokenizer(w=30,k=5,demo_mode=False)
text_tile = nltk.TextTilingTokenizer(w=30,k=5,demo_mode=True)
```

We will first focus our attention on the second object. Calling the `tokenize()` function with the first object on our preprocessed text will give us a list of paragraphs that have been grouped together in topical subsections. We can print items from this list to easily visualize the paragraphs that have been grouped together.

**Code:**
```python
sections = text_tile_chunk.tokenize(text)

for i in range(len(sections)):
    print(i+1,':',sections[i],'\n')
```
**Partial output:**
```

1 : If you look at a Map of the World, you will see, in the left-hand upper corner of the Eastern Hemisphere, two Islands lying in the sea.  They are England and Scotland, and Ireland.  England and Scotland form the greater part of these Islands.  Ireland is the next in size.  The little neighbouring islands, which are so small upon the Map as to be mere dots, are chiefly little bits of Scotland,--broken off, I dare say, in the course of a great length of time, by the power of the restless water.

 In the old days, a long, long while ago, before Our Saviour was born on earth and lay asleep in a manger, these Islands were in the same place, and the stormy sea roared round them, just as it roars now.  But the sea was not alive, then, with great ships and brave sailors, sailing to and from all parts of the world.  It was very lonely.  The Islands lay solitary, in the great expanse of water.  The foaming waves dashed against their cliffs, and the bleak winds blew over their forests; but the winds and waves brought no adventurers to land upon the Islands, and the savage Islanders knew nothing of the rest of the world, and the rest of the world knew nothing of them. 

2 : 

 It is supposed that the Phoenicians, who were an ancient people, famous for carrying on trade, came in ships to these Islands, and found that they produced tin and lead; both very useful things, as you know, and both produced to this very hour upon the sea-coast. The most celebrated tin mines in Cornwall are, still, close to the sea.  One of them, which I have seen, is so close to it that it is hollowed out underneath the ocean; and the miners say, that in stormy weather, when they are at work down in that deep place, they can hear the noise of the waves thundering above their heads.  So, the Phoenicians, coasting about the Islands, would come, without much difficulty, to where the tin and lead were. 

3 : 

 The Phoenicians traded with the Islanders for these metals, and gave the Islanders some other useful things in exchange.  The Islanders were, at first, poor savages, going almost naked, or only dressed in the rough skins of beasts, and staining their bodies, as other savages do, with coloured earths and the juices of plants.  But the Phoenicians, sailing over to the opposite coasts of France and Belgium, and saying to the people there, 'We have been to those white cliffs across the water, which you can see in fine weather, and from that country, which is called BRITAIN, we bring this tin and lead,' tempted some of the French and Belgians to come over also.  These people settled themselves on the south coast of England, which is now called Kent; and, although they were a rough people too, they taught the savage Britons some useful arts, and improved that part of the Islands.  It is probable that other people came over from Spain to Ireland, and settled there.

 Thus, by little and little, strangers became mixed with the Islanders, and the savage Britons grew into a wild, bold people; almost savage, still, especially in the interior of the country away from the sea where the foreign settlers seldom went; but hardy, brave, and strong. 
```

As we can see, the first three subsections formed include two two-paragraph subsections and one one-paragraph subsection. This may be judged a successful or a not successful parse depending on how it compares with human judgements, and we will discuss this in the following section. As most paragraphs are put in separate topic chunks, this may suggest that each paragraph is relatively independent in terms of its contents, relative to the document as a whole. Different parameter settings for `k` and `w` may yield different results.

Next we will turn our attention to the TextTiling object with the parameter setting of `demo_mode=True`. This will allow us to visualize the similarity between pseudosentences and see where the *valleys* reached the threshhold for section boundary detection.

```
#Code below adapted from NLTK TextTiling Demo (https://www.nltk.org/_modules/nltk/tokenize/texttiling.html#demo)

gap_scores, smoothed_g_scores, depth_scores, breaks = text_tile.tokenize(text)

pylab.xlabel("Sentence Gap index")
pylab.ylabel("Gap Scores")
pylab.plot(range(len(gap_scores)), gap_scores, label="Gap Scores")
pylab.plot(range(len(smoothed_g_scores)), smoothed_g_scores, label="Smoothed Gap scores")
pylab.plot(range(len(depth_scores)), depth_scores, label="Depth scores")
pylab.stem(range(len(breaks)), breaks, use_line_collection=True)
pylab.legend()
pylab.show()
```

The output of the above code will be a visualization of the text, giving depth scores and section boundaries.

![Demo Mode Output](https://github.com/veverica-hrdzava/discoursetutorial/blob/main/TextTiling.png)

Demo Mode will also allow us to produce a list of the boundaries between pseudosentences, noting those where a boundary was detected.

**Code:**
```python
print(breaks)

pseudo_breaks = []
for i in range(len(breaks)):
    if breaks[i] == 1:
        pseudo_breaks.append(i+1)
print(pseudo_breaks)
```
**Output**
```
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
[7, 15, 20, 26, 30, 38, 43, 47, 54, 62, 68, 73, 79, 83, 92, 98, 104, 108, 114, 119]
```
The first list has a value for each pseudosentence. A `0` indicates that it was not judged to be a subsection boundary, a `1` indicates that it was judged to be a subsection boundary. The second list simply gives the paragraph numbers corresponding to the subdivided text we previously printed.


## 3. Discourse Segmentation Evaluation: NLTK WindowDiff Tool

WindowDiff ([Pevzner and Hearst,2002](https://www.aclweb.org/anthology/J02-1002.pdf)) is a method that can be used to determine the success of the segmentation done by TextTiling. It is also available through [NLTK metrics](http://www.nltk.org/api/nltk.metrics.html?highlight=windowdiff#nltk.metrics.segmentation.windowdiff) and the source code can be found [here](http://www.nltk.org/_modules/nltk/metrics/segmentation.html#windowdiff).

To be able to compare the output of the TextTiling algorithm with another, we need to be able to access the pseudosentences so that they can be compared with human judgements or with other segmentation algorithms. This is not entirely straightforward, but can be easily accomplished with a few lines of code.

**Code:**
```python
seqs = text_tile_chunk._divide_to_tokensequences(text)

count = 1
for i in seqs:
    print('Pseudosentence #:',count)
    sequence = ''
    for i in i.__dict__['wrdindex_list']:
        sequence += i[0]+' '
    print(sequence)
    count += 1
```
**Partial Output:**
```
Pseudosentence #: 1
If you look at a Map of the World you will see in the left hand upper corner of the Eastern Hemisphere two Islands lying in the sea They are 
Pseudosentence #: 2
England and Scotland and Ireland England and Scotland form the greater part of these Islands Ireland is the next in size The little neighbouring islands which are so small upon 
Pseudosentence #: 3
the Map as to be mere dots are chiefly little bits of Scotland broken off I dare say in the course of a great length of time by the power 
Pseudosentence #: 4
of the restless water In the old days a long long while ago before Our Saviour was born on earth and lay asleep in a manger these Islands were in 
Pseudosentence #: 5
the same place and the stormy sea roared round them just as it roars now But the sea was not alive then with great ships and brave sailors sailing to 
```

Other segmentation can then be easily coded for comparison using WindowDiff. We will use the list of `1`s and `0`s representing pseudosentences that we produced earlier, along with a hypothetic human-annotated list. WindowDiff requires a string of `1`s and `0`s for calculation, so we will need to convert these to strings.

**Code:**
```python
text_tiling_segs = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
human_segs = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

tt_string = ''
h_string = ''
for i in text_tiling_segs:
    tt_string += str(i)
for i in human_segs:
    h_string += str(i)

print(tt_string)
print(h_string)
```
**Output:**
```
00000010000000100001000001000100000001000010001000000100000001000001000010000010001000000001000001000001000100000100001000000
01000010000000100001000000000100000001000010001000000100000001000001000011000110001000000001000001000001000100000100001000000
```

We will need to import the required library and then can run the comparison between the two segmented texts. This will give us a value between `0`, identical, and `1`, completly different. Obviously 

**Code:**

```python
from nltk.metrics import segmentation

segmentation.windowdiff(tt_string,h_string,3)
```
**Output:**

```
0.08943089430894309
```

## 4. Discussion


