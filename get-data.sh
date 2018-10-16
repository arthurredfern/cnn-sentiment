#!/bin/bash

echo "Downloading data..."
# Customer Review Datasets (5 products)
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
unzip CustomerReviewData.zip
# Move into data folder
mv "customer review data" raw
mv raw data/
# Remove downloaded zip
rm CustomerReviewData.zip

# Additional Customer Review Datasets (9 products)
wget http://www.cs.uic.edu/~liub/FBS/Reviews-9-products.rar
unar Reviews-9-products.rar
mv Reviews-9-products/* data/raw/
rm -r Reviews-9-products Reviews-9-products.rar

# More Customer Review Datasets (3 products)
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviews-3-domains.rar
unar CustomerReviews-3-domains.rar
mv "CustomerReviews -3 domains (IJCAI2015)"/*.txt  data/raw/
rm -r "CustomerReviews -3 domains (IJCAI2015)" CustomerReviews-3-domains.rar

# Download GloVe embeddings
echo "Downloading embeddings..."
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings
rm glove.6B.zip
