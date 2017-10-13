#!/bin/bash

if [ $OSTYPE == "cygwin" ] || [ $OSTYPE == "msys" ]; then
    cmd /c sphinx-apidoc -f -e -o source/ ../project05/
elif [[ $OSTYPE == "linux"* ]]; then
    sphinx-apidoc -f -e -o source/ ../project05/
fi
