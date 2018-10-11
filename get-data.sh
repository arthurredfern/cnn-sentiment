#!/bin/bash
# Get original zip file
wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
unzip CustomerReviewData.zip
# Move into data folder
mv "customer review data" raw
mv raw data/
# Remove downloaded zip
rm CustomerReviewData.zip
