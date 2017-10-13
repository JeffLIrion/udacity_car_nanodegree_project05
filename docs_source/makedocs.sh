#!/bin/bash

export _project_docs_source="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function sphinx_make() {
    if [ $OSTYPE == "cygwin" ] || [ $OSTYPE == "msys" ]; then
        cmd /c make.bat $1
    elif [[ $OSTYPE == "linux"* ]]; then
        make $1
    fi
}

cd $_project_docs_source
touch $_project_docs_source/source/*.*;
sphinx_make html;
if [ -d "$_project_docs_source/../docs" ]; then
    rm -rf $_project_docs_source/../docs
fi
mkdir -p $_project_docs_source/../docs;
if [ -d "$_project_docs_source/build/html" ]; then
    cp -r $_project_docs_source/build/html/* $_project_docs_source/../docs/;
fi
