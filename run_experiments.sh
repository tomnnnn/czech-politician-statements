#!/bin/sh

DIR="src/evaluate"
export PYTHONPATH=$PYTHONPATH:src

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

i=0
for file in "$DIR"/*.py; do
    base=$(basename "$file")
    name="${base%.py}"
    if [ "$base" != "__init__.py" ]; then
        i=$((i + 1))
        eval "FILE_$i='$name'"
        echo "$i) $name"
    fi
done

if [ "$i" -eq 0 ]; then
    echo "No experiments found in $DIR."
    exit 1
fi

# Prompt user
printf "Select an experiment to run (1-%d): " "$i"
read CHOICE

# Validate input
case "$CHOICE" in
    ''|*[!0-9]*) echo "Invalid input"; exit 1 ;;
    *) if [ "$CHOICE" -lt 1 ] || [ "$CHOICE" -gt "$i" ]; then
           echo "Invalid selection."; exit 1
       fi ;;
esac

# Get the module name and run
eval "SELECTED_NAME=\$FILE_$CHOICE"
MODULE_PATH="evaluate.$SELECTED_NAME"

echo "Running: python -m $MODULE_PATH"
python -m "$MODULE_PATH"

