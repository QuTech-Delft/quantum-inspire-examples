#!/bin/bash
#Collect all notebooks to be executed
echo "Running notebooks"
set -ev
export QI_UNITTEST=ON
find docs/notebooks/ -type f -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' file;
do
    jupyter nbconvert --to notebook --execute "$file"
done
