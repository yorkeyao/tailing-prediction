import csv

def is_num_by_except(num):
    try:
        float(num)
        return True
    except ValueError:
#        print "%s ValueError" % num
        return False

#print("bad".isalpha())

path = "/Users/zhangmengyan/Downloads/data-hydrosaver/"
file_in = path + "file_in.csv"
file_out_valid = path + "file_out_valid.csv"
file_out_invalid = path + "file_out_invalid.csv"


csv_reader = csv.reader(open(file_in))
csv_writer_valid = csv.writer(open(file_out_valid, 'wb'))
csv_writer_invalid = csv.writer(open(file_out_invalid, 'wb'))

rows = [row for row in csv_reader]
csv_writer_valid.writerow(rows[0])
csv_writer_invalid.writerow(rows[0])

for row in rows[1:]:
    flag = True
    for grid in row:
        if not is_num_by_except(grid):
            flag = False
            csv_writer_invalid.writerow(row)
            break
    if flag:
        csv_writer_valid.writerow(row)


