#!/bin/bash
# Get original zip file
echo "Downloading data..."
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
unzip CustomerReviewData.zip
# Move into data folder
mv "customer review data" raw
mv raw data/
# Remove downloaded zip
rm CustomerReviewData.zip

# Download GloVe embeddings
echo "Downloading embeddings..."
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings
rm glove.6B.zip
