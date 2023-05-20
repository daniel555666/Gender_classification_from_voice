# This file will run the data preprocessing scripts one by one, so Daniel won't have to do it manually.

# Run the data preprocessing scripts
# echo "Starting spanish."
# python3 data_process/emmissions_to_json.py spanish
# echo "Spanish done!"
# echo "Starting english."
# python3 data_process/emmissions_to_json.py english
# echo "English done!"
# echo "Starting french."
# python3 data_process/emmissions_to_json.py french
# echo "French done!"
# echo "Starting arabic."
# python3 data_process/emmissions_to_json.py arabic
# echo "Arabic done!"
# echo "Starting Russian."
# python3 data_process/emmissions_to_json.py russian
# echo "Russian done!"

# The same as in the last lines only in a loop
for lang in spanish english french arabic russian
do
    echo "Starting $lang."
    python3 data_process/emmissions_to_json.py $lang
    echo "$lang done!"
done