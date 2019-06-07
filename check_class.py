count = 0
with open('fraud_prep.csv') as infile:
    for line in infile:
        if line[-3] != '0':
            print(line[-3])
            count += 1
infile.close()

print(count)