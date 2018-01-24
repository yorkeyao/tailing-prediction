import datetime
import time
import csv

def string2timestamp(strValue):  
  
    try:          
        #d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S.%f") 
        d = datetime.datetime.strptime(strValue, "%d/%m/%y %H:%M")  
        t = d.timetuple()  
        timeStamp = int(time.mktime(t))  
        timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond))/1000000  
        #print timeStamp  
        return timeStamp  
    except ValueError as e:  
        print e  
        #d = datetime.datetime.strptime(str2, "%Y-%m-%d %H:%M:%S") 
        d = datetime.datetime.strptime(strValue, "%d/%m/%y %H:%M")  
        t = d.timetuple()  
        timeStamp = int(time.mktime(t))  
        timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond))/1000000  
        #print timeStamp  
        return timeStamp  

path = "/Users/zhangmengyan/Downloads/data-hydrosaver/"
file_in = path + "publishable_test_set.csv"
file_out = path + "file_out.csv"

csv_reader = csv.reader(open(file_in))
csv_writer = csv.writer(open(file_out, 'wb'))

rows = [row for row in csv_reader]

csv_writer.writerow(rows[0])

first_time_stamp = string2timestamp("1/4/15 0:00")
#count = 0 

for row in rows[1:]:
    row[0] = (string2timestamp(row[0]) - first_time_stamp)/ 60
    #print row[0]
    csv_writer.writerow(row)
    #count += 1

#print "count:"
#print count
    
    



