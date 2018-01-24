import csv
import sys

path = "/Users/zhangmengyan/Downloads/data-hydrosaver/"
file = path + "train_file_valid.csv"

stdout_backup=sys.stdout
log_file=open(path + "disruption.log" , "w")
sys.stdout=log_file

print "Extract disruption points (WQI8100XCL1.CPV Flotation Circuit Feed (tons per hour) decreases > 400):"

csv_reader = csv.reader(open(file))

rows = [row for row in csv_reader]

last_row = float(rows[1][1])

for row in rows[1:]:
    try:
        float(row[1])
    except  ValueError as e: 
        print row[0]
        print e

    if last_row - float(row[1]) > 400:
       print "timestamp " + row[0] +": change from "+ str(last_row) + " to " + str(float(row[1]))
    last_row = float(row[1])

log_file.close()
sys.stdout=stdout_backup
#print "log file successfully written!"

