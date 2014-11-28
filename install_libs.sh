#!/bin/tcsh

setenv PYTHONPATH ${HOME}:${HOME}/lib/python2.7/site-packages/
echo $PYTHONPATH

# Install networkx 1.9
wget https://pypi.python.org/packages/source/n/networkx/networkx-1.9.1.tar.gz#md5=a2d9ee8427c5636426f319968e0af9f2
tar -xzvf networkx-1.9.1.tar.gz
cd networkx-1.9.1
python setup.py install --prefix=${HOME}


cd ..
# Install Cython 0.21
wget http://cython.org/release/Cython-0.21.1.tar.gz
tar -xzvf Cython-0.21.1.tar.gz
cd Cython-0.21.1
python setup.py install --prefix=${HOME}

# Install scikit-image
cd ..
wget https://github.com/scikit-image/scikit-image/zipball/master
unzip master
cd scikit-image-scikit-image-*/
python setup.py install --prefix=${HOME}
