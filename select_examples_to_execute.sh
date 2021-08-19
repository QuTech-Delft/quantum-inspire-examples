#!/bin/bash
# Collect all python examples to be executed
echo "Running examples"
set -e -v
export QI_UNITTEST=ON
find docs/examples/ -type f -name "example_*.py" -print0 | while IFS= read -r -d $'\0' file;
do
    echo "running $file"
    python "$file"
done
