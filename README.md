# MIR Project
## Image content search using AlexNet and Glove.

The idea of this project was to: given a query **Q** retrieve a subset of images **I** that are related to the query in some way. <br>
### Pre-processing
All images are first passed to a pre trained AlexNet CNN, obtaining the **top n** classes for each one. <br>
Then I calculated each associated class **Glove** word vector and averaged them to generate an average word for all the classes related to the image. After this, using **Glove** I got the **top k** most similar words to this average word and store the relationship between each of this word with the current image in a hash table. <br>

### Making a Query

A query consist of two things:
- The query itself (text).
- Top n parameter.

So first I took each word in the query text, got its **Glove** word vector and average them to get an average query vector. <br>
Then with this average query vector I get the **top n** most similar words, we call this subset of words **Q**. <br>
Now for each **q** in **Q** : we get all the images associated to key **q** in the hast table built in the pre-processing part.

### The App

The application is built as a web application over Django (I recently learned Django and wanted some more hands on experience).