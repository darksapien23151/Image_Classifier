# Image_Classifier
Neural network to classify images based off ImageNet's data 

# Download the flower dataset from Udacity's S3 bucket
wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'

# Remove any existing "flowers" directory (if present)
rm -rf flowers

# Create a new "flowers" directory and extract the dataset
mkdir flowers && tar -xzf flower_data.tar.gz -C flowers

