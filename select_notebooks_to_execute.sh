#!/bin/bash
# Collect all notebooks to be executed
# classification_example1_2_data_points.ipynb is excluded because it takes more than 10 minutes (kills CI-pipeline)
echo "Running notebooks"
set -e
set -v
export QI_UNITTEST=ON
find docs/notebooks/ -type f -name "*.ipynb" ! -name 'classification_example1_2_data_points.ipynb' -print0 | while IFS= read -r -d $'\0' file;
do
    jupyter nbconvert --to notebook --execute "$file"
done
