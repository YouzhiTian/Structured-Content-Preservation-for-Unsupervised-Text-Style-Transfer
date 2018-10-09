# Sturctured-Content-Preservation-for-Unsupervised-Text-Style-Transfer
Authors: Youzhi Tian, Zhiting Hu, Zhou Yu

NOTES:
* The paper is under review at ICLR 2019 https://openreview.net/pdf?id=S1lCbhAqKX.
	* If you encounter any problems, please give comments on open review, or submit a pull request if you know the fix!
* This is research code meant to serve as a reference implementation. We do not recommend heavily extending or modifying this codebase for other purposes.


## Data Format
There are two kinds of dataset. The data has been preprocessed into <code>train_data.pkl</code> and <code>dev_data.pkl</code>. The batch size is 128 and the length of sentence is no more than 15.
* The <code>data/yelp/</code> directory contains Yelp review dataset used to transfer sentiment (positive and negative).
* The <code>data/political_data/</code> directory contains political dataset used to transfer political slant (democratic and republican).

## Quick Start
To train a model, run the following command:
```bash
python3 main.py
```
To test a model run the following command:
```bash
python3 main.py -if_eval True -file_save output.txt -checkpoint model_path -datapath data_path batch_size 1 
```
```
Where output.txt contains the transferred sentences of the test.merge of each dataset, model_path is the path of the model to be tested, data_path can be ./data/yelp/ and ./data/political_data/.
```

## Dependencies
Python == 3.6, Pytorch == 0.4.1, Torchtext == 0.2.3, GloVe:twitter 27B  <br>
Python requirement: numpy, pickle, argparse, nltk == 3.3.

## Results ##

Text style transfer is a task to transfer the style of the sentence (e.g., positive/negative sentiment) and keep the content of the original one.

We use automatic metrics to evaluate the output sentences: 
* We can use a pre-trained classifier to classify the generated sentences and evaluate the accuracy.
* We also evaluate the BLEU score between the generated sentences and the original sentences.
* We evaluate the BLEU (human) between the generated sentences and the human annotated sentences.
* We evaluate the POS distance between the generated sentences and the original sentences (the smaller the better).

The implementation here gives the following performance after 1 epoch of pre-training and 3 epochs of full-training:

| Accuracy (%)  | BLEU (with the original sentence) | BLEU (with the human annotated sentence)| POS distance |
| -------------------------------------| ----------------------------------| ----------------------------------|----------------------------------|
| 92.7 | 63.3  | 24.9 | 0.569 |

### Samples ###
Here are some randomly-picked samples. In each pair, the first sentence is the original sentence and the second is the generated. The whole samples can be seen in <code>./samples/output.txt</code>.
```
the happy hour crowd here can be fun on occasion.
the unhappy hour crowd here can be disappointed on occasion.

the menudo here is perfect.
the menudo here is nasty.

the service was excellent and my hostess was very nice and helpful.
the service was terrible and my hostess was very disappointing and unhelpful.

if you travel a lot do not stay at this hotel.
if you travel a lot do always stay at this hotel.

maria the manager is a horrible person.
maria the manager is a perfect person.

avoid if at all possible.
recommend if at all possible.

```
