# COS711_Assignment3
## Dependencies
* pandas
* xlsxwriter
* statistics
* tensorflow

## Usage
To run an experimaent, execute the following command in the terminal (assuming python and all the dependencies are installed) :

python .\train.py <spreadsheet_name>

where worksheet_name is the name of the spreadsheet that the average results of each eopch should be written to. This can also be omiiteed in which the current unix timestamp shall be used as the spreadsheet name.

To change the number of epochs or runs that should be executed, change the EPOCHS and NUM_RUNS variables respectively at the top of train.py
