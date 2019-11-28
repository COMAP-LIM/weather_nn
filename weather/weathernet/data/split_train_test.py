import sys
import random

if len(sys.argv) < 3:
    print("Usage: subset.py    'good subsequences'-filename    'bad subsequences'-filename    n (optional)")
    sys.exit()

good_obsids_file = sys.argv[1] 
bad_obsids_file = sys.argv[2]

f1 = open(good_obsids_file, 'r')
good_lines = f1.readlines()

f2 = open(bad_obsids_file, 'r')
bad_lines = f2.readlines()


if len(sys.argv) == 4:
    n = sys.argv[3]
else:
    n = len(bad_lines)

# Choose n random files    
good_lines_subset = random.sample(good_lines, n)
bad_lines_subset = random.sample(bad_lines, n)

for i in range(n):
    good_lines_subset[i] = good_lines_subset[i][:-2] + '    0\n'
    bad_lines_subset[i] = bad_lines_subset[i][:-2] + '    1\n'

all_lines = good_lines_subset + bad_lines_subset
random.shuffle(all_lines)

training_set = all_lines[:int(0.75*len(all_lines))]
testing_set = all_lines[int(0.75*len(all_lines)):]


f1 = open('training_data.txt', 'w')
f2 = open('testing_data.txt', 'w')
for i in range(len(training_set)):
    f1.write(training_set[i])

for i in range(len(testing_set)):
    f2.write(testing_set[i])
